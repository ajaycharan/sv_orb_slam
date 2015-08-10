#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <fstream>
#include <boost/thread.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/count.hpp>

#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>

#include "Tracking.h"
#include "FramePublisher.h"
#include "Map.h"
#include "MapPublisher.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Converter.h"

namespace ORB_SLAM {

using namespace boost::accumulators;

class Node {
 public:
  Node(const ros::NodeHandle& pnh);

  void run();
  void cameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                const sensor_msgs::CameraInfoConstPtr& cinfo_msg);
  void estimateFps();

 private:
  void loadVocabulary();

  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_camera_;
  accumulator_set<double, features<tag::count, tag::mean>> acc_;

  FramePublisher pub_frame_;
  ORBVocabulary orb_vocabulary_;
  Map world_;
  double fps_{30};
  bool fps_estimated_{false};
};

Node::Node(const ros::NodeHandle& pnh) : pnh_(pnh), it_(pnh) {
  loadVocabulary();
}

void Node::run() {
  // Create KeyFrame Database
  KeyFrameDatabase key_frame_db(orb_vocabulary_);

  pub_frame_.SetMap(&world_);

  // Create Map Publisher for Rviz
  MapPublisher pub_map(&world_);

  // Initialize the tracking Thread and launch
  Tracking tracker(&orb_vocabulary_, &pub_frame_, &pub_map, &world_, fps_);
  boost::thread tracking_thread(&Tracking::Run2, &tracker);
  tracker.SetKeyFrameDatabase(&key_frame_db);
  ROS_INFO("Initialize tracking thread");

  // Initialize the Local Mapping Thread and launch
  LocalMapping local_mapper(&world_);
  boost::thread local_mapping_thread(&LocalMapping::Run, &local_mapper);
  ROS_INFO("Initialize local mapping thread");

  // Initialize the Loop Closing Thread and launch
  LoopClosing loop_closer(&world_, &key_frame_db, &orb_vocabulary_);
  boost::thread loop_closing_thread(&LoopClosing::Run, &loop_closer);
  ROS_INFO("Initialize loop closing thread");

  // Set pointers between threads
  tracker.SetLocalMapper(&local_mapper);
  tracker.SetLoopClosing(&loop_closer);

  local_mapper.SetTracker(&tracker);
  local_mapper.SetLoopCloser(&loop_closer);

  loop_closer.SetTracker(&tracker);
  loop_closer.SetLocalMapper(&local_mapper);

  // Main thread
  ros::Rate rate(fps_);

  while (ros::ok()) {
    pub_frame_.Refresh();
    pub_map.Refresh();
    tracker.CheckResetByPublishers();
    rate.sleep();
  }

  // Save keyframe poses at the end of the execution
  ofstream f;

  vector<ORB_SLAM::KeyFrame*> keyframes = world_.GetAllKeyFrames();
  std::sort(keyframes.begin(), keyframes.end(), ORB_SLAM::KeyFrame::lId);

  ROS_INFO("Saving Keyframe Trajectory to KeyFrameTrajectory.txt");
  const auto strFile =
      ros::package::getPath("orb_slam") + "/" + "KeyFrameTrajectory.txt";
  f.open(strFile.c_str());
  f << std::fixed;

  for (ORB_SLAM::KeyFrame* kf : keyframes) {
    if (kf->isBad()) continue;

    cv::Mat R = kf->GetRotation().t();
    vector<float> q = ORB_SLAM::Converter::toQuaternion(R);
    cv::Mat t = kf->GetCameraCenter();
    f << setprecision(6) << kf->mTimeStamp << setprecision(7) << " "
      << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " "
      << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
  }
  f.close();
}

void Node::cameraCb(const sensor_msgs::ImageConstPtr& image_msg,
                    const sensor_msgs::CameraInfoConstPtr& cinfo_msg) {
  static double prev_time = 0.0;
  const auto curr_time = image_msg->header.stamp.toSec();
  if (prev_time != 0.0) {
    acc_(curr_time - prev_time);
  }
  prev_time = curr_time;
  if (extract_result<tag::count>(acc_) >= 5) {
    fps_ = std::round(1 / extract_result<tag::mean>(acc_));
    ROS_INFO("Estimated fps: %f", fps_);
    fps_estimated_ = true;
    sub_camera_.shutdown();
  }
}

void Node::estimateFps() {
  const auto image_topic = pnh_.resolveName("image");
  sub_camera_ = it_.subscribeCamera(image_topic, 1, &Node::cameraCb, this);
  ROS_INFO("Start estimating camera fps");
  while (!fps_estimated_) ros::spinOnce();
}

void Node::loadVocabulary() {
  std::string vocabulary_file;
  if (!pnh_.getParam("vocabulary", vocabulary_file)) {
    throw std::runtime_error("No vocabulary file.");
  }
  ROS_INFO("Loading vocabulary from disk. This may take a while.");
  const auto start = ros::Time::now();
  cv::FileStorage fs_voc(vocabulary_file.c_str(), cv::FileStorage::READ);
  if (!fs_voc.isOpened()) {
    throw std::runtime_error("Wrong path to vocabulary.");
  }
  orb_vocabulary_.load(fs_voc);
  const auto elapsed = ros::Time::now() - start;
  ROS_INFO("Vocabulary loaded from: %s, time: %f", vocabulary_file.c_str(),
           elapsed.toSec());
}

}  // namespace ORB_SLAM

int main(int argc, char** argv) {
  ros::init(argc, argv, "orb_slam");
  ros::NodeHandle pnh("~");

  ORB_SLAM::Node node(pnh);
  node.estimateFps();
  node.run();
}
