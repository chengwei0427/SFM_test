

#include "base.h"
#include <boost/graph/graph_concepts.hpp>


// computeKeyPointsAndDesp 同时提取关键点与特征描述子
void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor, int detector_size )
{/*
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

//     cv::initModule_nonfree();
    _detector = ORB::create(detector_size); // 提取1000个orb特征点
    _descriptor = ORB::create();
    //_detector = cv::FeatureDetector::create( detector.c_str() );
    //_descriptor = cv::DescriptorExtractor::create( descriptor.c_str() );

    if (!_detector || !_descriptor)
    {
        cerr << "Unknown detector or discriptor type !" << detector << "," << descriptor << endl;
        return;
    }

    _detector->detect( frame.rgb, frame.kp );
    _descriptor->compute( frame.rgb, frame.kp, frame.desp );*/
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 2, 0.04, 10);
    frame.kp.clear();
    sift->detectAndCompute(frame.rgb, noArray(), frame.kp, frame.desp);

    // 保存颜色信息
    vector<Vec3b> colors(frame.kp.size());
    for (int i = 0; i < frame.kp.size(); ++i)
    {
        Point2f& p = frame.kp[i].pt;
        colors[i] = frame.rgb.at<Vec3b>(p.y, p.x);
    }
    frame.colors = colors;

    return;
}

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure)
{
    //两个相机的投影矩阵[R T]，triangulatePoints只支持float型
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);

    R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
    T1.convertTo(proj1.col(3), CV_32FC1);

    R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T2.convertTo(proj2.col(3), CV_32FC1);

    Mat fK;
    K.convertTo(fK, CV_32FC1);
    proj1 = fK * proj1;
    proj2 = fK * proj2;

    //三角重建
    Mat s;
    cout << "triangulatePoints" <<endl;
    triangulatePoints(proj1, proj2, p1, p2, s);

    structure.clear();
    structure.reserve(s.cols);
    for (int i = 0; i < s.cols; ++i)
    {
        Mat_<float> col = s.col(i);
        col /= col(3);  //齐次坐标，需要除以最后一个元素才是真正的坐标值
        structure.push_back(Point3f(col(0), col(1), col(2)));
    }
}

bool checkKeyFrame(FRAME& lastFrame,FRAME& currFrame, RESULT_OF_PNP& result, int inlier_threshold, int KF_interval, double keyframe_threshold, double max_norm)
{
    if ( result.inliers < inlier_threshold )    //inliers不够，放弃该帧
           {
                   cout << "LESS INLIERS " << endl;
                   return false;
            }
    if ((currFrame.frameID - lastFrame.frameID) < KF_interval) // interval less 间隔比较小
               {
                   cout << " LESS INTERVAL " << endl;
                   return false;
               }
        // 计算运动范围是否太大
        double norm = normofTransform(result.rvec, result.tvec);
         cout << "NORM is : " << norm << endl;
       if ( norm >= max_norm ) 
       {
              cout << "TOO FAR AWAY " << endl;   // too far away, may be error 运动太大
               return false;
         }
       if ( norm <= keyframe_threshold )
       {
                   cout << "TOO_CLOSE" << endl;   // too adjacent frame 太近
                   return false;
         }
         
         return true;
}


// estimateMotion 计算两个帧之间的运动
// 输入：帧1和帧2
// 输出：rvec 和 tvec
RESULT_OF_PNP estimateMotion(
    FRAME& frame1, FRAME& frame2,
    vector<int>& struct_indices,
    vector<DMatch>& goodMatches,
    vector<Point3f>& points,
    CAMERA_INTRINSIC_PARAMETERS& camera )
{
    RESULT_OF_PNP result;
    //第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;
    vector<KeyPoint> key_points = frame2.kp;

    // 相机内参
    for (size_t i = 0; i < goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        int query_idx = goodMatches[i].queryIdx;
        int train_idx = goodMatches[i].trainIdx;

        int struct_idx = struct_indices[query_idx];
        if (struct_idx < 0) continue;

        pts_obj.push_back(points[struct_idx]);
        pts_img.push_back(key_points[train_idx].pt);
    }

    if (pts_obj.size() == 0 || pts_img.size() == 0)
    {
        result.inliers = -1;
        cout << "POINT or IMAGE size is 0" << endl;
        return result;
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cout << "pts_obj number is :" << pts_obj.size() << endl;
    cout << " pts_img number is : " << pts_img.size() << endl;
   cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers );
//     solvePnPRansac(pts_obj, pts_img, cameraMatrix, noArray(), rvec, tvec);

    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;
    cout << "inliers after solvePnPRansac : " << inliers.rows << endl;

    return result;
}


void get_matched_colors(
    FRAME& first,
    FRAME& second,
    vector<DMatch> matches,
    vector<Vec3b>& out_c1,
    vector<Vec3b>& out_c2
)
{
    vector<Vec3b> c1 = first.colors;
    vector<Vec3b> c2 = second.colors;
    out_c1.clear();
    out_c2.clear();
    for (int i = 0; i < matches.size(); ++i)
    {
        out_c1.push_back(c1[matches[i].queryIdx]);
        out_c2.push_back(c2[matches[i].trainIdx]);
    }
}

void get_matched_points(
    FRAME& first,
    FRAME& second,
    vector<DMatch> matches,
    vector<Point2f>& out_p1,
    vector<Point2f>& out_p2
)
{
    vector<KeyPoint> p1 = first.kp;
    vector<KeyPoint> p2 = second.kp;
    out_p1.clear();
    out_p2.clear();
    for (int i = 0; i < matches.size(); ++i)
    {
        out_p1.push_back(p1[matches[i].queryIdx].pt);
        out_p2.push_back(p2[matches[i].trainIdx].pt);
    }
}

void fusion_points(
    vector<DMatch>& matches,
    vector<int>& struct_indices,
    vector<int>& next_struct_indices,
    vector<Point3f>& structure,
    vector<Point3f>& next_structure,
    vector<Vec3b>& colors,
    vector<Vec3b>& next_colors
)
{
    cout << " fusion_points" << endl;
    for (int i = 0; i < matches.size(); ++i)
    {
        int query_idx = matches[i].queryIdx;
        int train_idx = matches[i].trainIdx;

        int struct_idx = struct_indices[query_idx];
        if (struct_idx >= 0) //若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
        {
//             cout << "continue" << endl;
            next_struct_indices[train_idx] = struct_idx;
            continue;
        }
//         cout << "--------  add a new point  -------" << endl;
        //若该点在空间中已经存在，将该点加入到结构中，且这对匹配点的空间点索引都为新加入的点的索引
        structure.push_back(next_structure[i]);
        colors.push_back(next_colors[i]);
        struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
    }
}

// normofTransform
// 输入： 旋转、平移矩阵
// 输出： 变换量
double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
}

void find_frame_matches(
    FRAME& first, FRAME& second,
    double good_match_threshold,
    std::vector< DMatch >& goodMatches)
{
    Mat descriptors_1 = first.desp;
    Mat descriptors_2 = second.desp;
    /*
    vector<DMatch> matches;
    BFMatcher matcher ( NORM_HAMMING );
    matcher.match ( descriptors_1, descriptors_2, matches );

    double minDis = 9999;
    for ( size_t i = 0; i < matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    cout << "min dis = " << minDis << endl;
    if ( minDis < 10 )
        minDis = 8;
    goodMatches.clear();
    for ( size_t i = 0; i < matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold * minDis )
            goodMatches.push_back( matches[i] );
    }*/
    
    vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

	//获取满足Ratio Test的最小匹配的距离
	float min_dist = 9999.9;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.7 * knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	goodMatches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//排除不满足Ratio Test的点和匹配距离过大的点
		if (
		    knn_matches[r][0].distance > 0.7 * knn_matches[r][1].distance ||
		    knn_matches[r][0].distance > 6 * max(min_dist, 10.0f)
		)
			continue;
		//保存匹配点
		goodMatches.push_back(knn_matches[r][0]);
	}
	
    Mat img;
    drawMatches ( first.rgb, first.kp, second.rgb, second.kp, goodMatches, img );
    imshow ( "匹配点对", img );
    waitKey(0);
}
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

//     Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1, keypoints_1 );
    detector->detect ( img_2, keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    cv::FlannBasedMatcher matcher;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher.match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 1.5 * min_dist, 60.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }

    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, match, img_match );
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_goodmatch );
    imshow ( "所有匹配点对", img_match );
    imshow ( "优化后匹配点对", img_goodmatch );
    waitKey(0);
}

void pose_estimation_2d2d (
    vector<Point2f>& points1,
    vector<Point2f>& points2,
    Mat& K,
    Mat& R, Mat& t , Mat& mask)
{

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    //cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //-- 计算本质矩阵
   //根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
    double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
    Point2d principal_point(K.at<double>(2), K.at<double>(5));
//     Point2d principal_point ( 325.1, 249.7 );               //相机主点, TUM dataset标定值
//     int focal_length = 521;                     //相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point, RANSAC, 0.999, 1.0, mask );
    double feasible_count = countNonZero(mask);
    cout << (int)feasible_count << " -in- " << points1.size() << endl;
    //cout << "essential_matrix is " << endl << essential_matrix << endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    //cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point , mask);
//     cout<<"R is "<<endl<<R<<endl;
//     cout<<"t is "<<endl<<t<<endl;
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
    vector<Vec3b> p1_copy = p1;
    p1.clear();

    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
            p1.push_back(p1_copy[i]);
    }
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
    vector<Point2f> p1_copy = p1;
    p1.clear();

    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
            p1.push_back(p1_copy[i]);
    }
}

void triangulation (
    vector<Point2f>& pts_1,
    vector<Point2f>& pts_2,
    const Mat& R, const Mat& t,
    vector< Point3f >& points )
{
    Mat T1 = (Mat_<float> (3, 4) <<
              1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);
    Mat T2 = (Mat_<float> (3, 4) <<
              R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
             );
    Mat pts_4d;
    cout << "before triangulatePoints " <<endl;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
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
    }
}

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
           (
               ( p.x - K.at<double>(0, 2) ) / K.at<double>(0, 0),
               ( p.y - K.at<double>(1, 2) ) / K.at<double>(1, 1)
           );
}


void Fundamental_RANSAC(
    FRAME& first, FRAME& second,
    std::vector< DMatch >& m_InlierMatches )
{
    Mat& img_1 = first.rgb;
    Mat& img_2 = second.rgb;
    vector<KeyPoint> keypoints_1 = first.kp;
    vector<KeyPoint> keypoints_2 = second.kp;
    Mat descriptors_1 = first.desp;
    Mat descriptors_2 = second.desp;


    //-- 对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    //cv::FlannBasedMatcher matcher; // orb不能用此计算
    BFMatcher matcher ( NORM_HAMMING );
    matcher.match ( descriptors_1, descriptors_2, matches );

    //-- 匹配点对筛选

// 分配空间
    int ptCount = (int)matches.size();
    Mat p1(ptCount, 2, CV_32F);
    Mat p2(ptCount, 2, CV_32F);
// 把Keypoint转换为Mat
    Point2f pt;
    for (int i = 0; i < ptCount; i++)
    {
        pt = keypoints_1[matches[i].queryIdx].pt;
        p1.at<float>(i, 0) = pt.x;
        p1.at<float>(i, 1) = pt.y;

        pt = keypoints_2[matches[i].trainIdx].pt;
        p2.at<float>(i, 0) = pt.x;
        p2.at<float>(i, 1) = pt.y;
    }
    /*  //-- 把匹配点转换为vector<Point2f>的形式
     vector<Point2f> p1;
     vector<Point2f> p2;

     for ( int i = 0; i < ( int ) matches.size(); i++ )
     {
         p1.push_back ( keypoints_1[matches[i].queryIdx].pt );
         p2.push_back ( keypoints_2[matches[i].trainIdx].pt );
     }
     */
// 用RANSAC方法计算F
    Mat m_Fundamental;
// 上面这个变量是基本矩阵
    vector<uchar> m_RANSACStatus;
// 上面这个变量已经定义过，用于存储RANSAC后每个点的状态
    m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC, 4); // 最后一个参数适用于LMeds和RANSAC，用于点到对极线的最大距离，默认为3

// 计算野点个数
    int OutlinerCount = 0;
    for (int i = 0; i < ptCount; i++)
    {
        if (m_RANSACStatus[i] == 0) // 状态为0表示野点
        {
            OutlinerCount++;
        }
    }

// 计算内点
    vector<Point2f> m_LeftInlier;
    vector<Point2f> m_RightInlier;
    m_InlierMatches.clear();
//    vector<DMatch> m_InlierMatches;
// 上面三个变量用于保存内点和匹配关系
    int InlinerCount = ptCount - OutlinerCount;
// m_InlierMatches.resize(InlinerCount);
    m_LeftInlier.resize(InlinerCount);
    m_RightInlier.resize(InlinerCount);
    InlinerCount = 0;
    for (int i = 0; i < ptCount; i++)
    {
        if (m_RANSACStatus[i] != 0)
        {
            m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
            m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
            m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
            m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
            DMatch m;
            m.queryIdx = InlinerCount;
            m.trainIdx = InlinerCount;
            m_InlierMatches.push_back(m);
//           m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
//           m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
            InlinerCount++;
        }
    }

// 把内点转换为drawMatches可以使用的格式
    vector<KeyPoint> key1(InlinerCount);
    vector<KeyPoint> key2(InlinerCount);
    KeyPoint::convert(m_LeftInlier, key1);
    KeyPoint::convert(m_RightInlier, key2);

// 显示计算F过后的内点匹配
//     Mat OutImage;
//      drawMatches(img_1, key1, img_2, key2, m_InlierMatches, OutImage);
//
//      imshow("match features after RANSAC", OutImage);
//      cvWaitKey( 0 );
}
