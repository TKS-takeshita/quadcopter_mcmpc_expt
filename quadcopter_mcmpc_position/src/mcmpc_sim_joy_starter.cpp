#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>

class MCMPCSimJoyStarter : public rclcpp::Node
{
public:
    MCMPCSimJoyStarter()
        : Node("mcmpc_sim_joy_starter")
    {
        joy_pub_ = create_publisher<sensor_msgs::msg::Joy>("/joy", 10);
        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&MCMPCSimJoyStarter::tick, this));
    }

private:
    void tick()
    {
        elapsed_ticks_++;

        sensor_msgs::msg::Joy joy;
        joy.axes.resize(8, 0.0f);
        joy.buttons.resize(11, 0);

        // First A press: arm/offboard + takeoff. Second A press: start MCMPC square.
        if (elapsed_ticks_ == 10 || elapsed_ticks_ == 30) {
            joy.buttons[0] = 1;
        }

        joy_pub_->publish(joy);
        if (elapsed_ticks_ > 35) {
            timer_->cancel();
            RCLCPP_INFO(get_logger(), "MCMPC simulator start sequence sent");
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::Joy>::SharedPtr joy_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    int elapsed_ticks_ = 0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MCMPCSimJoyStarter>());
    rclcpp::shutdown();
    return 0;
}
