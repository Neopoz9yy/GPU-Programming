#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

int equations;

float norm(float* vec) {
    float max = fabs(vec[0]);
    for (int i = 1; i < equations; i++) {
        if (max < fabs(vec[i])) max = fabs(vec[i]);
    }
    return max;
}

void accessors(std::vector<float> matr, std::vector<float> vec, int equations, float accuracy, int iterations, sycl::device device) {
    sycl::queue queue(device,
        [](sycl::exception_list list) {
            for (auto& e : list) {
                try { std::rethrow_exception(e); }
                catch (std::exception& e) { std::cout << "error: " << e.what() << std::endl; }
            }
        }, sycl::property::queue::enable_profiling());

    size_t time = 0.;
    std::vector<float>  accuracy_current(equations, .0f),
                        accuracy_next(equations, .0f),
                        diff(equations, .0f);

    {
        sycl::buffer<float> buf_matr(matr.data(), matr.size()),
                            buf_vec(vec.data(), vec.size()),
                            buf_current(accuracy_current.data(), accuracy_current.size()),
                            buf_next(accuracy_next.data(), accuracy_next.size()),
                            buf_diff(diff.data(), diff.size());
        
        for (int i = 0; i < iterations; i++) {
            sycl::event event = queue.submit(
                [&](sycl::handler& cgh) {
                    auto    acc_matr = buf_matr.get_access<sycl::access::mode::read>(cgh),
                            acc_vec = buf_vec.get_access<sycl::access::mode::read>(cgh),
                            acc_current = buf_current.get_access<sycl::access::mode::read>(cgh);
                    auto    acc_next = buf_next.get_access<sycl::access::mode::write>(cgh),
                            acc_diff = buf_diff.get_access<sycl::access::mode::write>(cgh);

                    cgh.parallel_for(sycl::range<1>(equations),
                        [=](sycl::id<1> id) {
                            acc_next[id] = acc_vec[id];
                            for (int j = 0; j < equations; j++) 
                                acc_next[id] -= acc_matr[j * equations + id] * acc_current[j];

                            acc_next[id] += acc_matr[id * equations + id] * acc_current[id];
                            acc_next[id] /= acc_matr[id * equations + id];
                            acc_diff[id] = acc_current[id] - acc_next[id];
                        }
                    );
                }
            );
            event.wait_and_throw();

            time += event.get_profiling_info<sycl::info::event_profiling::command_end>() - event.get_profiling_info<sycl::info::event_profiling::command_start>();
            std::swap(buf_current, buf_next);
            if (norm(buf_diff.get_host_access().get_pointer()) / norm(buf_next.get_host_access().get_pointer()) < accuracy) break;
        }
    }

    std::vector<float> acc;
    for (int i = 0; i < vec.size(); i++) {
        float multipl = 0;
        for (int j = 0; j < vec.size(); j++)
            multipl += matr[j * vec.size() + i] * accuracy_current[j];

        acc.push_back(multipl - vec[i]);
    }
    printf("[Accessors] Time: %f ms Accuracy: %.10f\n", time / 1e6, norm(&acc[0]));
}

void shared(std::vector<float> matr, std::vector<float> vec, int equations, float accuracy, int iterations, sycl::device device) {
    sycl::queue queue(device,
        [](sycl::exception_list list) {
            for (auto& e : list) {
                try { std::rethrow_exception(e); }
                catch (std::exception& e) { std::cout << "error: " << e.what() << std::endl; }
            }
        }, sycl::property::queue::enable_profiling());

    auto    shared_matr = sycl::malloc_shared<float>(equations * equations, queue),
            shared_vec = sycl::malloc_shared<float>(equations, queue),
            shared_current = sycl::malloc_shared<float>(equations, queue),
            shared_next = sycl::malloc_shared<float>(equations, queue),
            shared_diff = sycl::malloc_shared<float>(equations, queue);

    size_t time = 0;

    for (int i = 0; i < equations; i++) {
        shared_current[i] = .0f;
        shared_vec[i] = vec[i];
        for (int j = 0; j < equations; j++) 
            shared_matr[j * equations + i] = matr[j * equations + i];
    }

    for (int i = 0; i < iterations; i++) {
        sycl::event event = queue.parallel_for(sycl::range<1>(equations), 
            [=](sycl::id<1> id) {
                shared_next[id] = shared_vec[id];
                for (int j = 0; j < equations; j++) 
                    shared_next[id] -= shared_matr[j * equations + id] * shared_current[j];
            
                shared_next[id] += shared_matr[id * equations + id] * shared_current[id];
                shared_next[id] /= shared_matr[id * equations + id];
                shared_diff[id] = shared_current[id] - shared_next[id];
            }
        );
        event.wait_and_throw();

        time += event.get_profiling_info<sycl::info::event_profiling::command_end>() - event.get_profiling_info<sycl::info::event_profiling::command_start>();
        std::swap(shared_current, shared_next);
        if (norm(shared_diff) / norm(shared_next) < accuracy) break;
    }

    std::vector<float> acc;
    for (int i = 0; i < vec.size(); i++) {
        float multipl = 0;
        for (int j = 0; j < vec.size(); j++)
            multipl += matr[j * vec.size() + i] * shared_current[j];

        acc.push_back(multipl - vec[i]);
    }
    printf("[Shared] Time: %f ms Accuracy: %.10f\n", time / 1e6, norm(&acc[0]));

    sycl::free(shared_matr, queue);
    sycl::free(shared_vec, queue);
    sycl::free(shared_current, queue);
    sycl::free(shared_next, queue);
    sycl::free(shared_diff, queue);
}

void device(std::vector<float> matr, std::vector<float> vec, int equations, float accuracy, int iterations, sycl::device device) {
    sycl::queue queue(device,
        [](sycl::exception_list list) {
            for (auto& e : list) {
                try { std::rethrow_exception(e); }
                catch (std::exception& e) { std::cout << "error: " << e.what() << std::endl; }
            }
        }, sycl::property::queue::enable_profiling());

    std::vector<float>  accuracy_current(equations, .0f),
                        diff(equations, .0f);
    size_t time = 0;

    auto    dev_matr = sycl::malloc_device<float>(equations * equations, queue),
            dev_vec = sycl::malloc_device<float>(equations, queue),
            dev_current = sycl::malloc_device<float>(equations, queue),
            dev_next = sycl::malloc_device<float>(equations, queue),
            dev_diff = sycl::malloc_device<float>(equations, queue);

    queue.memcpy(dev_matr, matr.data(), equations * equations * sizeof(float)).wait();
    queue.memcpy(dev_vec, vec.data(), equations * sizeof(float)).wait();

    for (int i = 0; i < iterations; i++) {
        sycl::event event = queue.parallel_for(
            sycl::range<1>(equations), 
                [=](sycl::id<1> id) {
                    dev_next[id] = dev_vec[id];
                    for (int j = 0; j < equations; j++) 
                        dev_next[id] -= dev_matr[j * equations + id] * dev_current[j];
                    
                    dev_next[id] += dev_matr[id * equations + id] * dev_current[id];
                    dev_next[id] /= dev_matr[id * equations + id];
                    dev_diff[id] = dev_current[id] - dev_next[id];
                }
        );
        event.wait_and_throw();

        time += event.get_profiling_info<sycl::info::event_profiling::command_end>() - event.get_profiling_info<sycl::info::event_profiling::command_start>();

        std::swap(dev_current, dev_next);
        if (norm(&dev_diff[0]) / norm(&dev_next[0]) < accuracy) {
            queue.memcpy(diff.data(), dev_diff, equations * sizeof(float)).wait();
            queue.memcpy(accuracy_current.data(), dev_current, equations * sizeof(float)).wait();
            break;
        }
    }
    std::vector<float> acc;
    for (int i = 0; i < vec.size(); i++) {
        float multipl = 0;
        for (int j = 0; j < vec.size(); j++)
            multipl += matr[j * vec.size() + i] * accuracy_current[j];

        acc.push_back(multipl - vec[i]);
    }
    printf("[Device] Time: %f ms Accuracy: %.10f\n", time / 1e6, norm(&acc[0]));

    sycl::free(dev_matr, queue);
    sycl::free(dev_vec, queue);
    sycl::free(dev_current, queue);
    sycl::free(dev_next, queue);
    sycl::free(dev_diff, queue);
}

int main(int argc, char* argv[]) {
    equations = std::stoi(argv[1]);
    int iterations = std::stoi(argv[3]);
    float accuracy = std::stof(argv[2]);
    std::string device_type = static_cast<std::string>(argv[4]);

    sycl::device my_device = device_type == "gpu" ? sycl::device(sycl::default_selector{}) : sycl::device(sycl::cpu_selector{});
    printf("Target device : %s\n", my_device.get_info<sycl::info::device::name>().c_str());

    std::vector<float>  matr, 
                        vec;
    for (int i = 0; i < equations; i++) {
        vec.push_back(static_cast<float>(rand()) / (RAND_MAX) * 2 - 1);
        for (int j = 0; j < equations; j++)
            matr.push_back(i == j ? equations * 2 : static_cast<float>(rand()) / (RAND_MAX) * 2 - 1);
    }

    accessors(matr, vec, equations, accuracy, iterations, my_device);
    shared(matr, vec, equations, accuracy, iterations, my_device);
    device(matr, vec, equations, accuracy, iterations, my_device);

    return 0;
}