

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "base.h"


// 给定index，读取一帧数据
FRAME readFrame( int index, ParameterReader& pd );


int main( int argc, char** argv )
{

    ParameterReader pd;
    int startIndex  =   atoi( pd.getData( "start_index" ).c_str() ); // 开始序列
    int endIndex    =   atoi( pd.getData( "end_index"   ).c_str() );
    int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    double max_norm = atof( pd.getData("max_norm").c_str() );
    double min_norm = atof( pd.getData("min_norm").c_str() );
    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    int min_KF = atof( pd.getData("min_KF").c_str() );
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    bool mstate = false;  // 是否初始化

    // 所有的关键帧都放在了这里
    vector< FRAME > keyframes;
    vector<vector<int>> correspond_struct_idx; //保存第i副图像中第j个特征点对应的structure中点的索引
    correspond_struct_idx.clear();
    correspond_struct_idx.resize(100);//
    // initialize
    cout << "Initializing ..." << endl;
    int currIndex = startIndex; // 当前索引为currIndex

    string detector = pd.getData( "detector" );
    string descriptor = pd.getData( "descriptor" );

    Mat K = ( Mat_<double> ( 3, 3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    FRAME lastFrame;
    int num_t;
    vector<Mat> rotations;
    vector<Mat> motions;
    vector<Point3d> points;
    for ( currIndex = startIndex; currIndex < endIndex; currIndex++ )
    {
        if (!mstate)
        {
            cout << "初始化" << endl;
            mstate = true;
            // 读取两帧数据
            FRAME firstFrame = readFrame( currIndex, pd );
            FRAME secondFrame = readFrame(currIndex + 1, pd);
            computeKeyPointsAndDesp( firstFrame, detector, descriptor );
            computeKeyPointsAndDesp( secondFrame, detector, descriptor ); //提取特征
            keyframes.push_back( firstFrame ); // 第一帧数据加入keyFrame

            cout << "RANSAC" << endl;
            vector< cv::DMatch > matches;
            Fundamental_RANSAC(firstFrame, secondFrame, matches );
            cout << "一共找到了" << matches.size() << "组匹配点" << endl;
            //-- 估计两张图像间运动
            Mat R, t;
            pose_estimation_2d2d ( firstFrame.kp, secondFrame.kp, matches, R, t );

            //-- 三角化
            triangulation( firstFrame.kp, secondFrame.kp, matches, R, t, points );
            correspond_struct_idx[0].resize(firstFrame.kp.size(), -1);
            correspond_struct_idx[1].resize(secondFrame.kp.size(), -1);

            Mat R0 = Mat::eye(3, 3, CV_64FC1);
            Mat T0 = Mat::zeros(3, 1, CV_64FC1);
            //保存变换矩阵
            //将旋转向量转换为旋转矩阵
//      Rodrigues(r, R);
            rotations.push_back(R0);
            rotations.push_back(R);
            motions.push_back(T0);
            motions.push_back(t);
            // 填写头两幅图像的结构索引
            int idx = 0;
            for (int i = 0; i < matches.size(); ++i)
            {
                correspond_struct_idx[0][matches[i].queryIdx] = idx;
                correspond_struct_idx[1][matches[i].trainIdx] = idx;
                ++idx;
            }
            num_t = 1;
            lastFrame = secondFrame;
            currIndex = currIndex + 1;
        }
        else
        {
            cout << "Reading files " << currIndex << endl;
            FRAME currFrame = readFrame( currIndex, pd ); // 读取currFrame
            computeKeyPointsAndDesp( currFrame, detector, descriptor ); //提取特征

            // 比较f1 和 f2
            vector< cv::DMatch > matches;
            Fundamental_RANSAC(lastFrame, currFrame, matches );
	    cout<< " good match after RANSAC : " << matches.size() << endl;
            RESULT_OF_PNP result = estimateMotion( lastFrame, currFrame, correspond_struct_idx[num_t], matches, points, camera );
            if ( result.inliers < min_inliers )    //inliers不够，放弃该帧
            {
                cout << "LESS INLIERS " << endl;
                continue;
            }
            if ((currFrame.frameID - lastFrame.frameID) < min_KF) // interval less 间隔比较小
            {
                cout << " LESS INTERVAL " << endl;
                continue;
            }
            // 计算运动范围是否太大
            double norm = normofTransform(result.rvec, result.tvec);
            cout << "NORM is : " << norm << endl;
            if ( norm >= max_norm ) {
                cout << "TOO FAR AWAY " << endl;   // too far away, may be error 运动太大
                continue;
            }
            if ( norm <= keyframe_threshold ) {
                cout << "TOO_CLOSE" << endl;   // too adjacent frame 太近
                continue;
            }
            cout << " A NEW KEYFRAME" << endl;
            keyframes.push_back( currFrame );
            Mat R;
            //将旋转向量转换为旋转矩阵
            Rodrigues(result.rvec, R);
            //保存变换矩阵
            rotations.push_back(R);
            motions.push_back(result.tvec);

            vector<Point2f> p1, p2;

            get_matched_points(lastFrame, currFrame, matches, p1, p2);
            cout << p1.size() << " p1  p2 " << p2.size() << endl;
            // 根据之前求解的R,T进行三维重建
            vector<Point3d> next_points;

            reconstruct(
                K, rotations[num_t], motions[num_t],
                R, result.tvec, p1, p2, next_points);
            correspond_struct_idx[num_t + 1].resize(currFrame.kp.size(), -1);
            fusion_points(
                matches, correspond_struct_idx[num_t],
                correspond_struct_idx[num_t + 1], points, next_points);
            cout << " points size is : " << points.size() << endl;

            imshow("KEY FRAME", currFrame.rgb);
            cvWaitKey( 0 );
            lastFrame = currFrame;
            num_t++;
        }
    }
    // 新建一个点云
    PointCloud::Ptr pointCloud( new PointCloud );
    for (int i = 0; i < points.size(); i++)
    {
        PointT p;
        p.x = points[i].x;
        p.y = points[i].y;
        p.z = points[i].z;
        pointCloud->points.push_back( p );
    }
    pointCloud->is_dense = false;
    cout << "点云共有" << pointCloud->size() << "个点." << endl;
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud );
    return 0;
}

FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");

    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");

    stringstream ss;
    ss << rgbDir << index << rgbExt;
    string filename;
    ss >> filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss << depthDir << index << depthExt;
    ss >> filename;

    f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}


