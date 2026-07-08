#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <opencv2/dnn.hpp>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "httplib.h"

// Global shared frame buffer for MJPEG streaming
std::mutex frame_mutex;
std::vector<uchar> latest_jpeg;

// Global video source control
std::mutex source_mutex;
std::string current_source_type = "video"; // "video" or "camera"
std::string current_video_path = "solidYellowLeft.mp4";
int current_camera_index = 0;
bool source_changed = false;

// ONNX Runtime globals
const int input_width = 640;
const int input_height = 640;
const int num_channels = 3;

// Shared variables between threads
std::mutex state_mutex;
cv::Mat current_raw_frame;
cv::Mat current_mask;
cv::Mat current_da_mask;
bool new_frame_available = false;
auto last_lane_change_time = std::chrono::steady_clock::now() - std::chrono::hours(1);
auto out_of_lane_start = std::chrono::steady_clock::now();
bool is_out_of_lane = false;

// Telemetry globals
std::mutex tel_mutex;
float tel_fps = 0.0f;
int tel_frame_number = 0;
int tel_total_frames = 0;
float tel_lateral_offset_px = 0.0f;
float tel_lateral_offset_ratio = 0.0f;
bool tel_left_lane_detected = false;
bool tel_right_lane_detected = false;
std::string tel_ldw_status = "CENTERED";
std::string tel_ldw_severity = "SAFE";
std::string tel_lca_status = "CENTERED";
std::string tel_lca_severity = "NORMAL";
std::string tel_alert_message = "";
int tel_lane_width_px = 0;
float tel_lane_confidence = 0.0f;
float tel_processing_ms = 0.0f;
int global_server_port = 8080;
void inference_thread() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolop");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(12);
        
#ifdef _WIN32
        const wchar_t* model_path = L"yolop_static.onnx";
#else
        const char* model_path = "yolop_static.onnx";
#endif

        Ort::Session session(env, model_path, session_options);
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::vector<const char*> input_names = {"images"};
        std::vector<const char*> output_node_names = {
            "/model.24/m.0/Conv_output_0", 
            "/model.24/m.1/Conv_output_0", 
            "/model.24/m.2/Conv_output_0",
            "2233",
            "2379"
        };

        cv::KalmanFilter kf_left(3, 3, 0);
        cv::KalmanFilter kf_right(3, 3, 0);
        
        kf_left.transitionMatrix = (cv::Mat_<float>(3, 3) << 1,0,0, 0,1,0, 0,0,1);
        cv::setIdentity(kf_left.measurementMatrix);
        cv::setIdentity(kf_left.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(kf_left.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf_left.errorCovPost, cv::Scalar::all(1));

        kf_right.transitionMatrix = (cv::Mat_<float>(3, 3) << 1,0,0, 0,1,0, 0,0,1);
        cv::setIdentity(kf_right.measurementMatrix);
        cv::setIdentity(kf_right.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(kf_right.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf_right.errorCovPost, cv::Scalar::all(1));

        bool left_kf_initialized = false;
        bool right_kf_initialized = false;

        while (true) {
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                if (!new_frame_available) {
                    continue; // Busy wait (or could use condition variable, but busy wait is fine for now with small sleep)
                }
                frame = current_raw_frame.clone();
                new_frame_available = false;
            }
            if (frame.empty()) continue;

            auto t0 = std::chrono::steady_clock::now();

            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(input_width, input_height), cv::Scalar(0, 0, 0), true, false);

            float* blob_data = (float*)blob.data;
            size_t plane_size = input_width * input_height;
            const float mean[3] = {0.485f, 0.456f, 0.406f};
            const float std_dev[3] = {0.229f, 0.224f, 0.225f};
            for (int c = 0; c < 3; ++c) {
                for (size_t i = 0; i < plane_size; ++i) {
                    blob_data[c * plane_size + i] = (blob_data[c * plane_size + i] - mean[c]) / std_dev[c];
                }
            }

            auto t1 = std::chrono::steady_clock::now();
            auto start_inference = t1;
            
            std::vector<int64_t> input_shape = {1, 3, 640, 640};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob_data, 1 * 3 * 640 * 640, input_shape.data(), 4);
            
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_node_names.data(), 5);
            auto t2 = std::chrono::steady_clock::now();

            float* da_seg = output_tensors[3].GetTensorMutableData<float>();
            float* ll_seg = output_tensors[4].GetTensorMutableData<float>();
            size_t layer_size = 640 * 640;
            cv::Mat da_mask(640, 640, CV_8UC1, cv::Scalar(0));
            cv::Mat mask(640, 640, CV_8UC1, cv::Scalar(0));
            
            for (size_t i = 0; i < layer_size; ++i) {
                if (da_seg[layer_size + i] > da_seg[i]) da_mask.data[i] = 255;
                if (ll_seg[layer_size + i] > ll_seg[i]) mask.data[i] = 255;
            }

            // Manual Anchor Decoding for Static ONNX Model
            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            
            int strides[3] = {8, 16, 32};
            float anchors[3][3][2] = {
                {{3,9}, {5,11}, {4,20}},
                {{7,18}, {6,39}, {12,31}},
                {{19,50}, {38,81}, {68,157}}
            };
            
            auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

            for (int s = 0; s < 3; ++s) {
                int stride = strides[s];
                int grid_w = 640 / stride;
                int grid_h = 640 / stride;
                float* data = output_tensors[s].GetTensorMutableData<float>();
                int hw = grid_w * grid_h;

                for (int a = 0; a < 3; ++a) {
                    float anchor_w = anchors[s][a][0];
                    float anchor_h = anchors[s][a][1];
                    for (int y = 0; y < grid_h; ++y) {
                        for (int x = 0; x < grid_w; ++x) {
                            int base_idx = a * 6 * hw + y * grid_w + x;
                            float conf_raw = data[base_idx + 4 * hw];
                            float conf = sigmoid(conf_raw);
                            
                            float cls_raw = data[base_idx + 5 * hw];
                            float cls_conf = sigmoid(cls_raw);
                            
                            float final_conf = conf * cls_conf;

                            if (final_conf > 0.4f) {
                                float dx = sigmoid(data[base_idx + 0 * hw]);
                                float dy = sigmoid(data[base_idx + 1 * hw]);
                                float dw = sigmoid(data[base_idx + 2 * hw]);
                                float dh = sigmoid(data[base_idx + 3 * hw]);

                                float cx = (dx * 2.0f - 0.5f + x) * stride;
                                float cy = (dy * 2.0f - 0.5f + y) * stride;
                                float w = std::pow(dw * 2.0f, 2.0f) * anchor_w;
                                float h = std::pow(dh * 2.0f, 2.0f) * anchor_h;

                                int left = std::max(0, (int)(cx - w / 2));
                                int top = std::max(0, (int)(cy - h / 2));
                                boxes.push_back(cv::Rect(left, top, (int)w, (int)h));
                                confidences.push_back(final_conf);
                            }
                        }
                    }
                }
            }

            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.4f, 0.5f, nms_indices);

            int cutoff_y = 360; // Default horizon
            for (int idx : nms_indices) {
                cv::Rect box = boxes[idx];
                int bottom_y = box.y + box.height;
                int center_x = box.x + box.width / 2;
                
                // If the vehicle is generally in the center column of the image and is below the horizon
                if (center_x >= 220 && center_x <= 420 && bottom_y > cutoff_y) {
                    cutoff_y = bottom_y;
                }
            }
            cutoff_y = std::min(cutoff_y, 560); // Don't truncate lower than the ego hood

            // Ego-zone lane change detection
            int valid_lane_rows = 0;
            int min_center_x = 999;
            int max_center_x = -1;
            int max_full_width = 0; // Track massive horizontal paint bulges
            
            int global_min_paint_x = 999;
            int global_max_paint_x = -1;

            for (int y = 500; y < 640; ++y) {
                int ego_pixels = 0;
                int full_width_pixels = 0;
                long sum_x = 0;

                for (int x = 0; x < 640; ++x) {
                    if (mask.data[y * 640 + x] == 255) {
                        full_width_pixels++;
                        
                        if (x < global_min_paint_x) global_min_paint_x = x;
                        if (x > global_max_paint_x) global_max_paint_x = x;
                        
                        if (x >= 280 && x <= 360) {
                            ego_pixels++;
                            sum_x += x;
                        }
                    }
                }
                
                if (full_width_pixels > max_full_width) {
                    max_full_width = full_width_pixels;
                }
                
                if (ego_pixels > 0 && ego_pixels < 120 && full_width_pixels < 150) {
                    valid_lane_rows++;
                    int center_x = sum_x / ego_pixels;
                    if (center_x < min_center_x) min_center_x = center_x;
                    if (center_x > max_center_x) max_center_x = center_x;
                } else if (ego_pixels >= 120) {
                    // std::cout << "[DEBUG] Row " << y << " REJECTED: ego_pixels=" << ego_pixels << " (>=120)" << std::endl;
                } else if (full_width_pixels >= 150 && ego_pixels > 0) {
                    // std::cout << "[DEBUG] Row " << y << " REJECTED: full_width=" << full_width_pixels << " (>=150)" << std::endl;
                }
            }
            
            if (valid_lane_rows > 0) {
                std::cout << "[DEBUG] Frame - valid_lane_rows: " << valid_lane_rows 
                          << ", spread: " << (max_center_x - min_center_x) 
                          << ", max_full: " << max_full_width << std::endl;
            }
            
            // 1. Reject slanted horizontal lines that drift rapidly across the zone
            if (max_center_x != -1 && (max_center_x - min_center_x) > 150) {
                if (valid_lane_rows > 10) std::cout << "  -> REJECTED by slant (>150)" << std::endl;
                valid_lane_rows = 0;
            }
            
            // 2. Reject Crosswalks / Huge Text
            // A crosswalk spans the entire road. A lane change line, even if diagonal,
            // shouldn't spread more than 200 pixels globally across the bottom 140 rows.
            int global_spread = 0;
            if (global_max_paint_x != -1) {
                global_spread = global_max_paint_x - global_min_paint_x;
            }
            if (global_spread > 250) {
                if (valid_lane_rows > 0) std::cout << "  -> REJECTED by crosswalk spread (" << global_spread << " > 250)" << std::endl;
                valid_lane_rows = 0;
            }
            
            // 3. Fallback: Reject massive horizontal paint bulges (like crosswalks) anywhere on the screen
            if (max_full_width > 40) {
                if (valid_lane_rows > 10) std::cout << "  -> REJECTED by max_full_width (>40)" << std::endl;
                valid_lane_rows = 0;
            }
            
            if (valid_lane_rows > 4) {
                std::cout << "  -> ACCEPTED LANE CHANGE!" << std::endl;
            }
            
            // --- Out of Lane Detection ---
            long sum_center_offset = 0;
            int offset_count = 0;
            int lane_width_sum = 0;
            bool left_found_ever = false;
            bool right_found_ever = false;

            for (int y = 560; y < 600; ++y) {
                int left_x = -1, right_x = -1;
                for (int x = 319; x >= 0; --x) {
                    if (mask.data[y * 640 + x] == 255) { left_x = x; break; }
                }
                for (int x = 320; x < 640; ++x) {
                    if (mask.data[y * 640 + x] == 255) { right_x = x; break; }
                }
                
                if (left_x != -1) left_found_ever = true;
                if (right_x != -1) right_found_ever = true;
                
                // Only consider valid bounding pairs that are not horizontal blobs
                if (left_x != -1 && right_x != -1 && (right_x - left_x > 150)) {
                    int center = (left_x + right_x) / 2;
                    sum_center_offset += (center - 320);
                    lane_width_sum += (right_x - left_x);
                    offset_count++;
                }
            }

            if (global_server_port == 8080) {
                std::vector<cv::Point2f> left_pts, right_pts;
                
                int track_left = 319;
                int track_right = 320;
                
                // Initialize bottom positions robustly
                for (int x = 319; x >= 0; --x) {
                    if (mask.data[639 * 640 + x] == 255) { track_left = x; break; }
                }
                for (int x = 320; x < 640; ++x) {
                    if (mask.data[639 * 640 + x] == 255) { track_right = x; break; }
                }
                if (track_left == 319) track_left = 100;
                if (track_right == 320) track_right = 540;
                
                for (int y = 639; y >= cutoff_y; y -= 10) {
                    int found_left = -1;
                    int found_right = -1;
                    
                    int search_start_l = std::min(639, track_left + 60);
                    int search_end_l   = std::max(0, track_left - 60);
                    for (int x = search_start_l; x >= search_end_l; --x) {
                        if (mask.data[y * 640 + x] == 255) { found_left = x; break; }
                    }
                    
                    int search_start_r = std::max(0, track_right - 60);
                    int search_end_r   = std::min(639, track_right + 60);
                    for (int x = search_start_r; x <= search_end_r; ++x) {
                        if (mask.data[y * 640 + x] == 255) { found_right = x; break; }
                    }
                    
                    if (found_left != -1) {
                        left_pts.push_back(cv::Point2f(found_left, y));
                        track_left = found_left;
                    }
                    if (found_right != -1) {
                        right_pts.push_back(cv::Point2f(found_right, y));
                        track_right = found_right;
                    }
                }

                // 3. Polynomial fitting: x = a*ynorm^2 + b*ynorm + c
                // Wrap the solver in a RANSAC loop to reject outliers (crosswalks/intersections)
                auto fitPolyRANSAC = [](const std::vector<cv::Point2f>& pts, cv::Mat& best_coeffs) {
                    if (pts.size() < 4) return false;
                    
                    // Fallback to simple solve if we barely have enough points
                    if (pts.size() == 4) {
                        cv::Mat A(pts.size(), 3, CV_32F), B(pts.size(), 1, CV_32F);
                        for (size_t i = 0; i < pts.size(); ++i) {
                            float ynorm = (pts[i].y - 360.0f) / 280.0f;
                            A.at<float>(i, 0) = ynorm * ynorm; A.at<float>(i, 1) = ynorm; A.at<float>(i, 2) = 1.0f;
                            B.at<float>(i, 0) = pts[i].x;
                        }
                        return cv::solve(A, B, best_coeffs, cv::DECOMP_SVD);
                    }
                    
                    int iterations = 50;
                    float threshold = 15.0f; // pixel tolerance for inliers
                    int best_inlier_count = 0;
                    std::vector<int> best_inlier_indices;
                    
                    cv::RNG rng(12345);
                    
                    for (int it = 0; it < iterations; ++it) {
                        // Randomly pick 3 points
                        int idx1 = rng.uniform(0, (int)pts.size());
                        int idx2 = rng.uniform(0, (int)pts.size());
                        int idx3 = rng.uniform(0, (int)pts.size());
                        if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3) continue;
                        
                        cv::Mat A(3, 3, CV_32F), B(3, 1, CV_32F), coeffs;
                        auto set_row = [&](int r, int idx) {
                            float ynorm = (pts[idx].y - 360.0f) / 280.0f;
                            A.at<float>(r, 0) = ynorm * ynorm; A.at<float>(r, 1) = ynorm; A.at<float>(r, 2) = 1.0f;
                            B.at<float>(r, 0) = pts[idx].x;
                        };
                        set_row(0, idx1); set_row(1, idx2); set_row(2, idx3);
                        
                        if (!cv::solve(A, B, coeffs, cv::DECOMP_SVD)) continue;
                        
                        // Count inliers
                        int inliers = 0;
                        std::vector<int> current_inliers;
                        float a = coeffs.at<float>(0), b = coeffs.at<float>(1), c = coeffs.at<float>(2);
                        for (size_t i = 0; i < pts.size(); ++i) {
                            float ynorm = (pts[i].y - 360.0f) / 280.0f;
                            float pred_x = a * ynorm * ynorm + b * ynorm + c;
                            if (std::abs(pred_x - pts[i].x) < threshold) {
                                inliers++;
                                current_inliers.push_back(i);
                            }
                        }
                        
                        if (inliers > best_inlier_count) {
                            best_inlier_count = inliers;
                            best_inlier_indices = current_inliers;
                        }
                    }
                    
                    // Final fit using all inliers of the best model
                    if (best_inlier_count >= 4) {
                        cv::Mat A(best_inlier_count, 3, CV_32F), B(best_inlier_count, 1, CV_32F);
                        for (int i = 0; i < best_inlier_count; ++i) {
                            int idx = best_inlier_indices[i];
                            float ynorm = (pts[idx].y - 360.0f) / 280.0f;
                            A.at<float>(i, 0) = ynorm * ynorm; A.at<float>(i, 1) = ynorm; A.at<float>(i, 2) = 1.0f;
                            B.at<float>(i, 0) = pts[idx].x;
                        }
                        return cv::solve(A, B, best_coeffs, cv::DECOMP_SVD);
                    } else if (best_inlier_count == 3) {
                        // If we can only get 3 inliers, just use the exact fit for them
                        cv::Mat A(3, 3, CV_32F), B(3, 1, CV_32F);
                        for (int i = 0; i < 3; ++i) {
                            int idx = best_inlier_indices[i];
                            float ynorm = (pts[idx].y - 360.0f) / 280.0f;
                            A.at<float>(i, 0) = ynorm * ynorm; A.at<float>(i, 1) = ynorm; A.at<float>(i, 2) = 1.0f;
                            B.at<float>(i, 0) = pts[idx].x;
                        }
                        return cv::solve(A, B, best_coeffs, cv::DECOMP_SVD);
                    }
                    return false;
                };

                cv::Mat l_coeffs, r_coeffs;
                bool l_fit = fitPolyRANSAC(left_pts, l_coeffs);
                bool r_fit = fitPolyRANSAC(right_pts, r_coeffs);

                // 4. Correct KF and get Posteriori state
                cv::Mat l_post, r_post;
                if (left_kf_initialized) {
                    cv::Mat l_pred_prior = kf_left.predict();
                    if (l_fit) {
                        kf_left.correct(l_coeffs);
                        l_post = kf_left.statePost;
                    } else {
                        l_post = l_pred_prior;
                    }
                } else if (l_fit) {
                    kf_left.statePost = l_coeffs.clone();
                    left_kf_initialized = true;
                    l_post = kf_left.statePost;
                }

                if (right_kf_initialized) {
                    cv::Mat r_pred_prior = kf_right.predict();
                    if (r_fit) {
                        kf_right.correct(r_coeffs);
                        r_post = kf_right.statePost;
                    } else {
                        r_post = r_pred_prior;
                    }
                } else if (r_fit) {
                    kf_right.statePost = r_coeffs.clone();
                    right_kf_initialized = true;
                    r_post = kf_right.statePost;
                }

                da_mask = cv::Scalar(0); // clear original raw segmentation
                
                // 5. Render mathematically perfect polygon
                if (left_kf_initialized && right_kf_initialized) {
                    std::vector<cv::Point> poly_pts;
                    
                    // Left curve going UP from the bottom
                    for (int y = 639; y >= cutoff_y; y -= 5) {
                        float ynorm = (y - 360.0f) / 280.0f;
                        float xl = l_post.at<float>(0) * ynorm * ynorm + l_post.at<float>(1) * ynorm + l_post.at<float>(2);
                        float xr = r_post.at<float>(0) * ynorm * ynorm + r_post.at<float>(1) * ynorm + r_post.at<float>(2);
                        if (xl >= xr) break; // Embracing perspective convergence check
                        poly_pts.push_back(cv::Point(xl, y));
                    }
                    
                    // Right curve going DOWN to the bottom
                    std::vector<cv::Point> right_side;
                    for (int y = 639; y >= cutoff_y; y -= 5) {
                        float ynorm = (y - 360.0f) / 280.0f;
                        float xl = l_post.at<float>(0) * ynorm * ynorm + l_post.at<float>(1) * ynorm + l_post.at<float>(2);
                        float xr = r_post.at<float>(0) * ynorm * ynorm + r_post.at<float>(1) * ynorm + r_post.at<float>(2);
                        if (xl >= xr) break; // Embracing perspective convergence check
                        right_side.push_back(cv::Point(xr, y));
                    }
                    std::reverse(right_side.begin(), right_side.end());
                    poly_pts.insert(poly_pts.end(), right_side.begin(), right_side.end());
                    
                    if (poly_pts.size() > 2) {
                        std::vector<std::vector<cv::Point>> polys = {poly_pts};
                        cv::fillPoly(da_mask, polys, cv::Scalar(255));
                    }
                }
            }

            cv::Mat mask_resized;
            cv::resize(mask, mask_resized, frame.size(), 0, 0, cv::INTER_NEAREST);
            cv::Mat da_mask_resized;
            cv::resize(da_mask, da_mask_resized, frame.size(), 0, 0, cv::INTER_NEAREST);

            auto t3 = std::chrono::steady_clock::now();
            std::cout << "[PROFILE] Preprocess: " << std::chrono::duration<float, std::milli>(t1 - t0).count() << "ms | "
                      << "Inference: " << std::chrono::duration<float, std::milli>(t2 - t1).count() << "ms | "
                      << "Postprocess/Fusion: " << std::chrono::duration<float, std::milli>(t3 - t2).count() << "ms" << std::endl;


            {
                std::lock_guard<std::mutex> lock(state_mutex);
                current_mask = mask_resized;
                current_da_mask = da_mask_resized;
                auto now = std::chrono::steady_clock::now();

                // 1. Lane Change Logic
                // If offset_count is high, it means we clearly see BOTH the left and right lane boundaries
                // forming a coherent lane (which happens during a turn). 
                // We should only trigger a lane change if we are crossing the line (which destroys the bounding pairs).
                if (valid_lane_rows > 4 && offset_count < 10) {
                    last_lane_change_time = now;
                    is_out_of_lane = false; // Reset "out of lane" during active lane change
                }

                // 2. Out of Lane Logic
                if (offset_count > 10 && valid_lane_rows <= 4) {
                    int avg_offset = sum_center_offset / offset_count;
                    if (std::abs(avg_offset) > 50) { // Off center by 50+ pixels
                        if (!is_out_of_lane && (now - out_of_lane_start > std::chrono::seconds(2))) {
                            is_out_of_lane = true;
                        }
                    } else {
                        // Centered properly
                        out_of_lane_start = now;
                        is_out_of_lane = false;
                    }
                } else {
                    // Lines missing OR actively changing lanes
                    out_of_lane_start = now;
                }

                // Update Telemetry
                {
                    std::lock_guard<std::mutex> tlock(tel_mutex);
                    tel_processing_ms = std::chrono::duration<float, std::milli>(now - start_inference).count();
                    tel_left_lane_detected = left_found_ever;
                    tel_right_lane_detected = right_found_ever;
                    tel_lane_confidence = std::min(1.0f, valid_lane_rows / 40.0f);
                    if (offset_count > 0) {
                        tel_lane_width_px = lane_width_sum / offset_count;
                    }
                    if (offset_count > 0) {
                        tel_lateral_offset_px = (float)sum_center_offset / offset_count;
                        tel_lateral_offset_ratio = tel_lateral_offset_px / 320.0f;
                    }
                    
                    bool is_changing = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_lane_change_time).count() < 2000;
                    
                    if (is_out_of_lane) {
                        tel_alert_message = "OUT OF LANE";
                        tel_ldw_status = "OUT OF LANE";
                        tel_ldw_severity = "WARNING";
                        tel_lca_status = "CENTERED";
                        tel_lca_severity = "NORMAL";
                    } else if (is_changing) {
                        tel_alert_message = "LANE CHANGED";
                        tel_lca_status = "LANE CHANGED";
                        tel_lca_severity = "NOTICE";
                        tel_ldw_status = "CENTERED";
                        tel_ldw_severity = "SAFE";
                    } else {
                        tel_alert_message = "";
                        tel_ldw_status = "CENTERED";
                        tel_ldw_severity = "SAFE";
                        tel_lca_status = "CENTERED";
                        tel_lca_severity = "NORMAL";
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in inference_thread: " << e.what() << "\n" << std::flush;
    } catch (...) {
        std::cerr << "Unknown exception in inference_thread.\n";
    }
}

void capture_thread() {
    cv::VideoCapture cap;
    
    auto open_source = [&]() {
        if (cap.isOpened()) cap.release();
        if (current_source_type == "video") {
            cap.open(current_video_path);
        }
        else cap.open(current_camera_index);
        source_changed = false;
    };

    {
        std::lock_guard<std::mutex> lock(source_mutex);
        open_source();
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> lock(source_mutex);
            if (source_changed) open_source();
        }

        cv::Mat orig_frame;
        cap >> orig_frame;
        if (orig_frame.empty()) {
            std::lock_guard<std::mutex> lock(source_mutex);
            if (current_source_type == "video") cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        cv::Mat frame;
        cv::resize(orig_frame, frame, cv::Size(854, 480));

        cv::Mat mask_to_draw;
        cv::Mat da_mask_to_draw;
        bool is_changing = false;
        bool out_of_lane_warning = false;
        
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            current_raw_frame = frame.clone();
            new_frame_available = true;
            if (!current_mask.empty()) mask_to_draw = current_mask.clone();
            if (!current_da_mask.empty()) da_mask_to_draw = current_da_mask.clone();
            
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_lane_change_time).count() < 2000) {
                is_changing = true;
            }
            out_of_lane_warning = is_out_of_lane;
        }

        if (!da_mask_to_draw.empty()) {
            cv::Mat overlay = frame.clone();
            overlay.setTo(cv::Scalar(0, 255, 0), da_mask_to_draw); // Drivable area in Green
            cv::addWeighted(overlay, 0.4, frame, 0.6, 0, frame);
        }

        if (!mask_to_draw.empty()) {
            frame.setTo(cv::Scalar(0, 0, 255), mask_to_draw); // Red lines on the edges
        }
        
        if (is_changing) {
            cv::putText(frame, "LANE CHANGE DETECTED!", cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        } else if (out_of_lane_warning) {
            cv::putText(frame, "PLEASE CENTER VEHICLE", cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 165, 255), 3, cv::LINE_AA); // Orange color
        }

        std::vector<uchar> buf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 60};
        cv::imencode(".jpg", frame, buf, params);

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_jpeg = std::move(buf);
        }

        frame_count++;
        int total_f = 0;
        if (current_source_type == "video") total_f = cap.get(cv::CAP_PROP_FRAME_COUNT);

        float current_fps = 0.0f;
        if (frame_count % 30 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            current_fps = 30.0f / std::chrono::duration<float>(current_time - start_time).count();
            std::cout << "Stream FPS: " << current_fps << std::endl;
            start_time = current_time;
        }

        {
            std::lock_guard<std::mutex> lock(tel_mutex);
            tel_frame_number = (current_source_type == "video") ? cap.get(cv::CAP_PROP_POS_FRAMES) : frame_count;
            tel_total_frames = total_f;
            if (current_fps > 0.0f) tel_fps = current_fps;
        }
        
        // Dynamic sleep to target 30 FPS
        auto frame_end = std::chrono::high_resolution_clock::now();
        int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        int sleep_time = std::max(1, 33 - elapsed_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
}

int main(int argc, char** argv) {
    // Force OpenCV to distribute its heavy preprocessing and JPEG compression across all 12 CPU cores
    cv::setNumThreads(12);

    int port = 8080;
    if (argc >= 2) {
        current_video_path = argv[1];
    }
    if (argc >= 3) {
        port = std::stoi(argv[2]);
    }
    global_server_port = port;

    std::cout << "Starting threads for " << current_video_path << " on port " << port << "...\n" << std::flush;
    std::thread t_capture(capture_thread);
    std::thread t_inference(inference_thread);

    std::cout << "Starting web server on port " << port << "...\n" << std::flush;
    httplib::Server svr;

    svr.Get("/video_feed", [](const httplib::Request&, httplib::Response& res) {
        std::string boundary = "frame";
        res.set_content_provider(
            "multipart/x-mixed-replace; boundary=" + boundary,
            [boundary](size_t offset, httplib::DataSink& sink) {
                std::vector<uchar> jpeg;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    jpeg = latest_jpeg;
                }
                if (!jpeg.empty()) {
                    std::string header = "--" + boundary + "\r\n"
                        "Content-Type: image/jpeg\r\n"
                        "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";
                    sink.write(header.data(), header.size());
                    sink.write(reinterpret_cast<const char*>(jpeg.data()), jpeg.size());
                    sink.write("\r\n", 2);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20)); // cap at 50fps max to avoid spinning
                return true;
            },
            [](bool) {}
        );
    });

    svr.Post("/switch_source", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(source_mutex);
        if (req.has_param("type")) {
            std::string type = req.get_param_value("type");
            if (type == "video" || type == "camera") {
                current_source_type = type;
                source_changed = true;
                res.set_content("{\"status\":\"ok\"}", "application/json");
                return;
            }
        }
        res.status = 400;
        res.set_content("{\"error\":\"Invalid parameters\"}", "application/json");
    });

    svr.Get("/events", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        
        res.set_content_provider(
            "text/event-stream",
            [](size_t, httplib::DataSink& sink) {
                std::string fps, frame, t_frames, offset, ratio, left_det, right_det, width, conf, proc;
                std::string ldw_st, ldw_sev, lca_st, lca_sev, msg;
                {
                    std::lock_guard<std::mutex> lock(tel_mutex);
                    fps = std::to_string(tel_fps);
                    frame = std::to_string(tel_frame_number);
                    t_frames = std::to_string(tel_total_frames);
                    offset = std::to_string(tel_lateral_offset_px);
                    ratio = std::to_string(tel_lateral_offset_ratio);
                    left_det = tel_left_lane_detected ? "true" : "false";
                    right_det = tel_right_lane_detected ? "true" : "false";
                    width = std::to_string(tel_lane_width_px);
                    conf = std::to_string(tel_lane_confidence);
                    proc = std::to_string(tel_processing_ms);
                    ldw_st = tel_ldw_status;
                    ldw_sev = tel_ldw_severity;
                    lca_st = tel_lca_status;
                    lca_sev = tel_lca_severity;
                    msg = tel_alert_message;
                }

                std::string json = "{";
                json += "\"fps\": " + fps + ", ";
                json += "\"frame_number\": " + frame + ", ";
                json += "\"total_frames\": " + t_frames + ", ";
                json += "\"lateral_offset_px\": " + offset + ", ";
                json += "\"lateral_offset_ratio\": " + ratio + ", ";
                json += "\"left_lane_detected\": " + left_det + ", ";
                json += "\"right_lane_detected\": " + right_det + ", ";
                json += "\"lane_width_px\": " + width + ", ";
                json += "\"lane_confidence\": " + conf + ", ";
                json += "\"processing_ms\": " + proc + ", ";
                json += "\"ldw_status\": \"" + ldw_st + "\", ";
                json += "\"ldw_severity\": \"" + ldw_sev + "\", ";
                json += "\"lca_status\": \"" + lca_st + "\", ";
                json += "\"lca_severity\": \"" + lca_sev + "\", ";
                json += "\"alert_message\": \"" + msg + "\"";
                json += "}";

                std::string event = "data: " + json + "\n\n";
                sink.write(event.data(), event.size());
                std::this_thread::sleep_for(std::chrono::milliseconds(200)); // send 5 updates per second
                return true;
            },
            [](bool) {}
        );
    });

    // Add CORS headers to all responses
    svr.set_post_routing_handler([](const auto&, auto& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    // Serve frontend static files
    svr.set_mount_point("/", "frontend");

    svr.listen("0.0.0.0", port);
    t_inference.join();
    return 0;
}
