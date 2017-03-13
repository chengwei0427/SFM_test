
# pragma once

// 各种头文件
// C++标准库
#include <fstream>
#include <vector>
#include <map>
using namespace std;

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/features2d/features2d.hpp"
//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
// #include <opencv2/xfeatures2d.hpp>



// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>


// 类型定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
using namespace std;
using namespace cv;
// Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
// 相机内参结构
struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx, cy, fx, fy, scale;
};

// 帧结构
struct FRAME
{
    int frameID;
    cv::Mat rgb; //该帧对应的彩色图
    cv::Mat desp;       //特征描述子
    vector<Vec3b> colors; // 保存特征点颜色
    vector<cv::KeyPoint> kp; //关键点
};

// PnP 结果
struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    int inliers;
};

// 函数接口

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );
// void Fundamental_RANSAC(
//     const Mat& img_1, const Mat& img_2,
//     std::vector<KeyPoint>& keypoints_1,
//     std::vector<KeyPoint>& keypoints_2,
//     std::vector< DMatch >& matches );
void Fundamental_RANSAC(
    FRAME& first, FRAME& second,
    std::vector<DMatch>& matches);
void pose_estimation_2d2d (
    vector<Point2f>& points1,
    vector<Point2f>& points2,
    Mat& K,
    Mat& R, Mat& t , Mat& mask);

// 像素坐标转相机归一化坐标
Point2f pixel2cam( const Point2d& p, const Mat& K );
void maskout_colors(vector<Vec3b>& p1, Mat& mask);
void maskout_points(vector<Point2f>& p1, Mat& mask);

void triangulation (
    vector<Point2f>& pts_1,
    vector<Point2f>& pts_2,
    const Mat& R, const Mat& t,
    vector< Point3f >& points );

//  计算旋转平移量，估计运动大小
double normofTransform( cv::Mat rvec, cv::Mat tvec );

// point2dTo3d 将单个点从图像坐标转换为空间坐标
// input: 3维点Point3f (u,v,d)
cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera );

// computeKeyPointsAndDesp 同时提取关键点与特征描述子
void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor , int detector_size);

void find_frame_matches(
    FRAME& first, FRAME& second,
    double good_match_threshold,
    std::vector< DMatch >& goodMatches);

void get_matched_colors(
    FRAME& first,
    FRAME& second,
    vector<DMatch> matches,
    vector<Vec3b>& out_c1,
    vector<Vec3b>& out_c2
);
void get_matched_points(
    FRAME& first,
    FRAME& second,
    vector<DMatch> matches,
    vector<Point2f>& out_p1,
    vector<Point2f>& out_p2
);

void fusion_points(
    vector<DMatch>& matches,
    vector<int>& struct_indices,
    vector<int>& next_struct_indices,
    vector<Point3f>& structure,
    vector<Point3f>& next_structure,
    vector<Vec3b>& colors,
    vector<Vec3b>& next_colors
);

void reconstruct(
    Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2,
    vector<Point2f>& p1, vector<Point2f>& p2,
    vector<Point3f>& structure);
//  checkKeyFrame 判断是否为关键帧
//  输入： 上一帧，当前帧，相机运动，inliers阈值，帧间隔， 最小运动，最大运动
bool checkKeyFrame(FRAME& lastFrame,FRAME& currFrame, RESULT_OF_PNP& result, int inlier_threshold, int KF_interval, double normofTF_min, double normofTF_max);

// estimateMotion 计算两个帧之间的运动
// 输入：帧1和帧2, 相机内参
RESULT_OF_PNP estimateMotion(
    FRAME& frame1, FRAME& frame2,
    vector<int>& struct_indices,
    vector<DMatch>& goodMatches,
    vector<Point3f>& points,
    CAMERA_INTRINSIC_PARAMETERS& camera );
// RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera );

// cvMat2Eigen
// Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec );

// 参数读取类
class ParameterReader
{
public:
    ParameterReader( string filename = "./parameters.txt" )
    {
        ifstream fin( filename.c_str() );
        if (!fin)
        {
            cerr << "parameter file does not exist." << endl;
            return;
        }
        while (!fin.eof())
        {
            string str;
            getline( fin, str );
            if (str[0] == '#')
            {
                // 以‘＃’开头的是注释
                continue;
            }

            int pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr( 0, pos );
            string value = str.substr( pos + 1, str.length() );
            data[key] = value;

            if ( !fin.good() )
                break;
        }
    }
    string getData( string key )
    {
        map<string, string>::iterator iter = data.find(key);
        if (iter == data.end())
        {
            cerr << "Parameter name " << key << " not found!" << endl;
            return string("NOT_FOUND");
        }
        return iter->second;
    }
public:
    map<string, string> data;
};

inline static CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
    return camera;
}


