#include <CL/sycl.hpp>
#include <vector>

int main(int argc, char* argv[]) {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    int devices_size, platforms_size = platforms.size();
    
    for (int i = 0; i < platforms_size; i++) {
        printf("Platform #%d : %s\n", i, platforms[i].get_info<sycl::info::platform::name>().c_str());
        std::vector<sycl::device> devices = platforms[i].get_devices();
        devices_size = devices.size();
        for (int j = 0; j < devices_size; j++) printf("  Device #%d : %s\n", j, devices[j].get_info<sycl::info::device::name>().c_str());
    }

    printf("\n");

    for (int i = 0; i < platforms_size; i++) {
        std::vector<sycl::device> devices = platforms[i].get_devices();
        for (int j = 0; j < devices_size; j++) {
            printf("%s\n", devices[j].get_info<sycl::info::device::name>().c_str());
            {
                sycl::buffer<int, 1> buffer_platform(&i, 1);
                sycl::buffer<int, 1> buffer_device(&j, 1);

                sycl::queue queue(devices[j]);

                queue.submit(
                    [&](sycl::handler& cgh) {
                        auto platform = buffer_platform.get_access<sycl::access::mode::read>(cgh);
                        auto device   = buffer_device.get_access<sycl::access::mode::read>(cgh);
                        sycl::stream s(1024, 80, cgh);

                        cgh.parallel_for(sycl::range<1>(4),
                            [=](sycl::id<1> id) {
                                s << id << ": Hello from platform " << platform[0] << " and device " << device[0] << sycl::endl;
                            }
                        );
                    }
                ).wait();
            }
        }
    }
    return 0;
}
