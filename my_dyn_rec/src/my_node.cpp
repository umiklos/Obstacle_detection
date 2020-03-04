#include <ros/ros.h>

#include <dynamic_reconfigure/server.h>
#include <my_dyn_rec/MyParamsConfig.h>

void callback(my_dyn_rec::MyParamsConfig &config, uint32_t level){
    
}

int main(int argc, char **argv){
    ros::init(argc,argv,"my_node");

    dynamic_reconfigure::Server<my_dyn_rec::MyParamsConfig> server;
    dynamic_reconfigure::Server<my_dyn_rec::MyParamsConfig>::CallbackType f;

    f=boost::bind(&callback,_1,_2);
    server.setCallback(f);

    ros::spin();

    return 0;
}