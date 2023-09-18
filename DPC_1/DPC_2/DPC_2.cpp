#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#define type double

int main(int argc, char* argv[]) {
    const std::string device_type = std::string(argv[2]);
    const int   intervals   = std::stoi(argv[1]),
                GROUP_SIZE  = 16,
                GROUP_COUNT = 16;
    sycl::device device = device_type == "gpu" ? sycl::device(sycl::default_selector{}) : sycl::device(sycl::cpu_selector{});
    std::vector<type> integral(GROUP_COUNT * GROUP_COUNT, 0.0f);

    try {
        sycl::buffer<type, 1> buffer(integral.data(), integral.size());

        sycl::property_list props{ sycl::property::queue::enable_profiling() };
        sycl::queue queue(device,
            [](sycl::exception_list list) {
                for (auto &e : list) {
                    try { std::rethrow_exception(e); }
                    catch (std::exception& e) {
                        std::cout << "error: " << e.what() << std::endl;
                        std::terminate();
                    }
                }
            }, props);

        printf("Number of rectangles : %d X %d\n", intervals, intervals);
        printf("Target device : %s\n", queue.get_device().get_info<sycl::info::device::name>().c_str());

        sycl::event event = queue.submit(
            [&](sycl::handler& cgh) {
                auto out = buffer.get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<class Integral>(
                    sycl::nd_range<2>(sycl::range<2>(GROUP_COUNT * GROUP_SIZE, GROUP_COUNT * GROUP_SIZE), sycl::range<2>(GROUP_SIZE, GROUP_SIZE)),
                    [=](sycl::nd_item<2> item) {
                        type sum = 0;
                        for(    type x = (item.get_global_id(0) + item.get_global_id(0) + 1.) / 2 / intervals; x <= 1; x += static_cast<type>(item.get_global_range(0)) / intervals)
                            for(type y = (item.get_global_id(1) + item.get_global_id(1) + 1.) / 2 / intervals; y <= 1; y += static_cast<type>(item.get_global_range(1)) / intervals)
                                sum += sin(x) * cos(y) / intervals / intervals;

                        type reduce = reduce_over_group(item.get_group(), sum, std::plus<type>());
                        
                        if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0)
                            out[item.get_group(0) + item.get_group(1) * item.get_group_range(0)] = reduce;
                    }
                );
            }
        );
        event.wait_and_throw();

        uint64_t    start   = event.get_profiling_info<sycl::info::event_profiling::command_start>(),
                    end     = event.get_profiling_info<sycl::info::event_profiling::command_end>();

        printf("Kernel execution time: %f ms\n", (end - start) / 1e6);
    }
    catch (sycl::exception& e) { std::cout << "error: " << e.what() << std::endl; }

    type    computed = 0,
            expected = -sin(1) * cos(1) + sin(1);
    for (type i : integral) computed += i;

    printf("Expected: %.10f\n", expected);
    printf("Computed: %.10f\n", computed);
    printf("Difference: %.10f\n", std::abs(expected - computed));

    return 0;
}