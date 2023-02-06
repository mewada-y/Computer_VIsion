#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>
#include <random>

using namespace cv;
using namespace std;

void get_img_value(Mat image){
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cout << "(";
            for (int k = 0; k < image.channels(); k++) {
                cout << (int)image.at<Vec3b>(i, j)[k] << ",";
            }
            cout << ") ";
        }
        cout << endl;
    }
}
void noisy_img_gen(){
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 2.0);

    Mat base_image(256, 256, CV_8UC1, Scalar(128));

    for (int i = 0; i < 10; i++) {
      Mat noisy_image = base_image.clone();
      for (int r = 0; r < 256; r++) {
        for (int c = 0; c < 256; c++) {
          int noise = (int)distribution(generator);
          noisy_image.at<uchar>(r, c) = saturate_cast<uchar>(noisy_image.at<uchar>(r, c) + noise);
        }
      }
      // Save the noisy image
      String filename = "noisy_image_" + to_string(i) + ".jpg";
      imwrite(filename, noisy_image);
    }
}
void EST_NOISE(){
    // Vec3b bgrPixel[256][256] = {};
    // double sigma[256][256] = {0};
    Mat bgrPixel(256, 256, CV_64F, 0.0);
    Mat sigma(bgrPixel.rows, bgrPixel.cols, CV_32F);
    
    // cout << sigma[69][69] << endl;
    for(uint8_t s = 0;s<10;s++){
        string file_name = "image_" + to_string(s) + ".jpg";
        Mat image = imread(file_name, IMREAD_COLOR);
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                bgrPixel.at<Vec3f>(i,j) += image.at<Vec3b>(i, j);
                // std::cout << bgrPixel[i][j] << ",";
                // for (int k = 0; k < image.channels(); k++) {
                //     bgrPixel[i][j] += (double)image.at<Vec3b>(i, j)[k];
                // }
            }
        }
    }
    for (int i = 0; i < bgrPixel.rows; i++) {
        for (int j = 0; j < bgrPixel.cols; j++) {
            bgrPixel.at<Vec3f>(i,j) /= 10;
            // cout << bgrPixel.at<Vec3f>(i, j) << endl;
        }
    }
    for(uint8_t s = 0;s<10;s++){
        string file_name = "image_" + to_string(s) + ".jpg";
        Mat image = imread(file_name, IMREAD_COLOR);
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                Vec3b img_b = image.at<Vec3b>(i, j);
                Vec3f img_flt;
                img_flt[0] = img_b[0];
                img_flt[1] = img_b[1];
                img_flt[2] = img_b[2];
                Vec3f diff = bgrPixel.at<Vec3f>(i, j) - img_flt;
                sigma.at<Vec3f>(i, j) += diff.mul(diff);
                // sigma.at<Vec3f>(i,j) += abs(bgrPixel.at<Vec3f>(i,j)-image.at<Vec3b>(i, j));
                //  std::cout << sigma[i][j] << ",";
                // for (int k = 0; k < image.channels(); k++) {
                //     sigma.at<Vec3f>(i,j) += minus(bgrPixel.at<Vec3f>(i,j) - image.at<Vec3b>(i, j)[k])=;
                // }
            }
        }
    }
    // int sigma_est = sum(sigma);
    sigma /= (sigma.rows * sigma.rows * 9);
    sqrt(sigma,sigma);
    cout << sum(sigma)/(sigma.rows * sigma.rows) << endl;
    // for (int i = 0; i < sigma.rows; i++) {
    //     for (int j = 0; j < sigma.cols; j++) {
    //         // accumulate(sigma.begin(),sigma.end(),0);
    //         sum(sigma);
    //         // sigma_est += sigma.at<Vec3f>(i);
    //         // cout << bgrPixel.at<Vec3f>(i, j) << endl;
    //     }
    // }
    // for (int i = 0; i < sigma.rows; i++) {
    //     for (int j = 0; j < sigma.cols; j++) {
    //         sigma.at<Vec3f>(i, j) /= 9*sigma.rows*sigma.rows;
    //         // cout << bgrPixel.at<Vec3f>(i, j) << endl;
    //     }
    // }
    cout << sigma.at<Vec3f>(128,128) << endl;
     // double std = 0.0;
     // for (int i = 0; i < 256; i++) {
     //     for (int j = 0; j < 256; j++) {
     //         sigma[i][j] += sqrt(sigma[i][j]/9);
     //         // std::cout << sigma[i][j] << ",";
             
     //     }
     // }
    // std::cout << sqrt(sigma/9) << ",";
    // return bgrPixel;
}
void box_filter(){
    Mat kernel = Mat::ones(3, 3, CV_32F) / 9;
    Mat filteredImage;
    for(uint8_t s = 0;s<10;s++){
        string file_name = "image_" + to_string(s) + ".jpg";
        Mat image = imread(file_name);
        filter2D(image, filteredImage, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
        imwrite("filtered_image.jpg", filteredImage);
        for(uint8_t i = 0;i<10;i++){
            string file_name = "filter_image_" + to_string(i) + ".jpg";
            imwrite(file_name,image);    
        }
    }
}
void gen_gaussian_mask(){
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1.4);
    double mask[8][8] = {0};
    for (int i = 0; i < 8; i++) {
        for (int r = 0; r < 8; r++) {
            double noise = (double)distribution(generator);
            mask[i][r] = noise;
            cout << mask[i][r] << endl;
      }
    }
}
int main()
{
    // int width = 256;
    // int height = 256;
    // const double mean = 0.0;
    // const double stde = 2.0;
    // Mat img(width,height,CV_8UC1,Scalar(128));
    // Mat noise(img.size(),img.type());
    // randn(noise,mean,stde);
    // imwrite("test.jpg",img);
    // img += noise;
    // // int blue = result[1];
    // printf("%lf",result);
    // get_img_value(img);
    // if (img.empty()){
    //     printf("No image was created");
    // }
    // for(uint8_t i = 0;i<10;i++){
    //     string file_name = "raw_image_" + to_string(i) + ".jpg";
    //     imwrite(file_name,img);    
    // }
    // noisy_img_gen();
    // EST_NOISE();
    gen_gaussian_mask();
    // box_filter();
    // for(uint8_t s = 0;s<10;s++){
    //     string file_name = "image_" + to_string(s) + ".jpg";  
    //     Mat image = imread(file_name);
    //     for (int i = 0; i < image.rows; i++) {
    //         for (int j = 0; j < image.cols; j++) {
    //             for (int k = 0; k < image.channels(); k++) {
    //                 cout << (double)image.at<Vec3b>(i, j)[k] << endl;
    //             }
    //         }
    //     }
    // }
    // // namedWindow("Temp",cv::WINDOW_AUTOSIZE);
    // // imshow("Temp",img);
    // waitKey(0);
    return 0;   
}