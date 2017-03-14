

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
    double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    int min_KF = atof( pd.getData("min_KF").c_str() );
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    bool mstate = false;  // 是否初始化
    int after_tri = 3; // 是否第一次3D-2D

    // 所有的关键帧都放在了这里
    vector< FRAME > keyframes;
    vector<vector<int>> correspond_struct_idx; //保存第i副图像中第j个特征点对应的structure中点的索引
    correspond_struct_idx.clear();
    correspond_struct_idx.resize(100);//
    // initialize
    cout << "Initializing ..." << endl;
    int currIndex = startIndex; // 当前索引为currIndex
    // 默认使用ORB
    string detector = pd.getData( "detector" );
    string descriptor = pd.getData( "descriptor" );
    int detector_size = atoi( pd.getData("detector_size").c_str() );

//     Mat K = ( Mat_<double> ( 3, 3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    Mat K = ( Mat_<double> ( 3, 3 ) << camera.fx, 0, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1 );

    FRAME lastFrame;
    int num_t;
    vector<Mat> rotations;
    vector<Mat> motions;
    vector<Point3f> points;
    vector<Vec3b> colors; // 颜色
    for ( currIndex = startIndex; currIndex <= endIndex; currIndex++ )
    {
        if (!mstate)
        {
            cout << "初始化" << endl;
            mstate = true;
            vector<Vec3b> c2;
            // 读取两帧数据
            FRAME firstFrame = readFrame( currIndex, pd );
            FRAME secondFrame = readFrame(currIndex + 1, pd);
            computeKeyPointsAndDesp( firstFrame, detector, descriptor , detector_size);
            computeKeyPointsAndDesp( secondFrame, detector, descriptor, detector_size ); //提取特征
            keyframes.push_back( firstFrame ); // 第一帧数据加入keyFrame

            cout << "RANSAC" << endl;
            vector< cv::DMatch > matches;
           // Fundamental_RANSAC(firstFrame, secondFrame, matches );
	    find_frame_matches(firstFrame, secondFrame, good_match_threshold, matches);
            cout << "一共找到了" << matches.size() << "组匹配点" << endl;
            get_matched_colors(firstFrame, secondFrame, matches, colors, c2);
	    vector<Point2f> pts_1, pts_2;
            get_matched_points(firstFrame, secondFrame, matches, pts_1, pts_2);
	    /*vector<Point2f> pts_1, pts_2;
           for ( DMatch m : matches )
           {
             // 将像素坐标转换至相机坐标
             pts_1.push_back ( pixel2cam( firstFrame.kp[m.queryIdx].pt, K) );  // 为什么不是这样？？？
             pts_2.push_back ( pixel2cam( secondFrame.kp[m.trainIdx].pt, K) );
            }*/
            cout << "pts_1 size : " << pts_1.size() << endl;
	        //-- 估计两张图像间运动
            Mat R, t;
            Mat mask;
            pose_estimation_2d2d ( pts_1, pts_2, K, R, t, mask );
	    
            maskout_colors(colors, mask);
           maskout_points(pts_1, mask);
           maskout_points(pts_2, mask);
           
	   Mat R0 = Mat::eye(3, 3, CV_64FC1);// 单位阵
            Mat t0 = Mat::zeros(3, 1, CV_64FC1);
	     //两个相机的投影矩阵[R T]，triangulatePoints只支持float型
             Mat T1(3, 4, CV_32FC1);
            Mat T2(3, 4, CV_32FC1);

           R0.convertTo(T1(Range(0, 3), Range(0, 3)), CV_32FC1);
           t0.convertTo(T1.col(3), CV_32FC1);

          R.convertTo(T2(Range(0, 3), Range(0, 3)), CV_32FC1);
          
	  t.convertTo(T2.col(3), CV_32FC1);
          Mat fK;
           K.convertTo(fK, CV_32FC1);
           T1 = fK * T1;  // 为什么要这样，还不懂？？？
           T2 = fK * T2;
	      //-- 三角化
             Mat pts_4d;
            cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
	    
	    	  
	       Mat TT1 = (Mat_<float> (3, 4) <<
              1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);
             Mat TT2 = (Mat_<float> (3, 4) <<
              R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
             );
	     vector<Point2f> PTS1,PTS2;
	     for(int i = 0;i < pts_1.size(); i++)
	     {
// 	       cout << pts_1[i] <<endl;
	       PTS1.push_back(pixel2cam(pts_1[i], K));
	       PTS2.push_back(pixel2cam(pts_2[i], K));
	    }
	     Mat PTS4d;
	     vector<Point3f> structure;
	     cv::triangulatePoints(TT1,TT2, PTS1, PTS2, PTS4d);
	     cout << "pts_4d" << pts_4d.cols << " --" << "PTS4d" << PTS4d.cols <<endl;
             // 转换成非齐次坐标
    for ( int i = 0; i < pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        Point3f p (
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0)
        );
        points.push_back( p );
	
	Mat y = PTS4d.col(i);
	y /= y.at<float>(3, 0);
	Point3f pp(
	  y.at<float>(0,0),
          y.at<float>(1,0),
          y.at<float>(2,0)
	);
	structure.push_back(pp);
    }
    
        PointCloud::Ptr cloud1( new PointCloud );
    for (int i = 0; i < points.size(); i++)
    {
        PointT p;
        p.x = points[i].x;
        p.y = points[i].y;
        p.z = points[i].z;
        p.b = colors[i][0];
        p.g = colors[i][1];
        p.r = colors[i][2];
        cloud1->points.push_back( p );
    }
    cloud1->is_dense = false;
    cout << "点云共有" << cloud1->size() << "个点." << endl;
    pcl::io::savePCDFileBinary("cloud1.pcd", *cloud1 );
    
	PointCloud::Ptr cloud2( new PointCloud);
	    for (int i = 0; i < structure.size(); i++)
    {
        PointT p;
        p.x = structure[i].x;
        p.y = structure[i].y;
        p.z = structure[i].z;
        p.b = colors[i][0];
        p.g = colors[i][1];
        p.r = colors[i][2];
        cloud2->points.push_back( p );
    }
    cloud2->is_dense = false;
    cout << "点云共有" << cloud2->size() << "个点." << endl;
    pcl::io::savePCDFileBinary("cloud2.pcd", *cloud2 );
    
//             triangulation(  pts_1, pts_2, R, t, points );

    //保存变换矩阵
            rotations.push_back(R0);
            rotations.push_back(R);
            motions.push_back(t0);
            motions.push_back(t);
            // 填写头两幅图像的结构索引
            correspond_struct_idx[0].resize(firstFrame.kp.size(), -1);
            correspond_struct_idx[1].resize(secondFrame.kp.size(), -1);
            int idx = 0;
            for (int i = 0; i < matches.size(); ++i)
            {
                if (mask.at<uchar>(i) == 0)
                    continue;
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
            cout << "-------------------------------------------------" << endl;
            cout << "Reading files " << currIndex << endl;
            FRAME currFrame = readFrame( currIndex, pd ); // 读取currFrame
            computeKeyPointsAndDesp( currFrame, detector, descriptor, detector_size ); //提取特征

            // 比较f1 和 f2
            vector< cv::DMatch > matches;
            // Fundamental_RANSAC(lastFrame, currFrame, matches );
            find_frame_matches(lastFrame, currFrame, good_match_threshold, matches);
            cout << " good match after RANSAC : " << matches.size() << endl;
            RESULT_OF_PNP result = estimateMotion( lastFrame, currFrame, correspond_struct_idx[num_t], matches, points, camera );
	    bool is_keyframe = checkKeyFrame(lastFrame, currFrame, result, min_inliers,min_KF, min_norm, max_norm);
	    if(!is_keyframe) 
	    {
	      cout << "------ NOT A KEYFRAME---------" <<endl;
	      if(after_tri>0)
		after_tri--;
	      else
	           continue;
	    }
            cout << "----- A NEW KEYFRAME---------" << endl;
            keyframes.push_back( currFrame );
	    cout << "keyframes size is : " << keyframes.size() << endl;
            Mat R;
            //将旋转向量转换为旋转矩阵
            Rodrigues(result.rvec, R);
            //保存变换矩阵
            rotations.push_back(R);
            motions.push_back(result.tvec);

            vector<Point2f> p1, p2;
            vector<Vec3b> c1, c2;
            get_matched_points(lastFrame, currFrame, matches, p1, p2);
            get_matched_colors(lastFrame, currFrame, matches, c1, c2);
            cout << p1.size() << " p1  p2 " << p2.size() << endl;
            // 根据之前求解的R,T进行三维重建
            vector<Point3f> next_points;

            reconstruct(
                K, rotations[num_t], motions[num_t],
                R, result.tvec, p1, p2, next_points);
            correspond_struct_idx[num_t + 1].resize(currFrame.kp.size(), -1);
            fusion_points(
                matches, correspond_struct_idx[num_t],
                correspond_struct_idx[num_t + 1],
                points, next_points,
                colors, c1);
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
        p.b = colors[i][0];
        p.g = colors[i][1];
        p.r = colors[i][2];
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

//     f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}


