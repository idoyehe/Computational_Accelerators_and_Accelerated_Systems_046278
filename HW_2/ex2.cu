/* compile with: nvcc -O3 -maxrregcount=32 ex2.cu -o ex2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <string.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_DIMENSION 32
#define NREQUESTS 10000
#define N_STREAMS 64
#define INVALID -1
#define VALID 1
#define Q_SLOTS 10
#define MAX_REGISTER_COUNT 32
#define SQR(a) ((a) * (a))
#define PIXEL_VALUES 256
#define SHARED_MEM_PER_BLOCK 3 * PIXEL_VALUES

#define INCREASE_PC_POINTER(X) ((X) + 1) % Q_SLOTS

typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

bool is_queue_full(int *p_id, int *c_id) {
    __sync_synchronize();
    return INCREASE_PC_POINTER(*p_id) == *c_id;
}

void enqueueJob(uchar *queue, int *p_id, int image_idx,uchar * images_in, int image_size, uchar valid){
    if(valid == VALID){
        memcpy(queue + (((*p_id) * (1 + image_size)) + 1), images_in + (image_idx * image_size), image_size);
    }
    memcpy(queue + (*p_id * ( 1 + image_size)), &valid, sizeof(uchar));
    __sync_synchronize();
    *p_id = INCREASE_PC_POINTER(*p_id);
    __sync_synchronize();
}

void dequeueJob(uchar *queue, int c_id, int image_idx, uchar *images_out, int image_size){
    memcpy(images_out + (image_idx * image_size), queue + (c_id * image_size), image_size);
    __sync_synchronize();
}

int numOfThreadBlocksCalc(int threadsPerBlock) {
    int sharedMemPerBlock = SHARED_MEM_PER_BLOCK;
    int regsPerBlock = threadsPerBlock * MAX_REGISTER_COUNT;

    cudaDeviceProp currDeviceProperties;

    CUDA_CHECK(cudaGetDeviceProperties(&currDeviceProperties, 0));
    // hardware limitation
    int numOfBlocksPerSMSharedMem = currDeviceProperties.sharedMemPerMultiprocessor / sharedMemPerBlock;
    int numOfBlocksPerSMRegs = currDeviceProperties.regsPerMultiprocessor / regsPerBlock;
    int numOfBlocksPerSMThreads = currDeviceProperties.maxThreadsPerMultiProcessor / threadsPerBlock;

    //Get the minimum threadBlock amount per multiProcessor subject to hardware limitation
    int minBlocksPerSM = numOfBlocksPerSMSharedMem;
    if (numOfBlocksPerSMRegs < minBlocksPerSM) minBlocksPerSM = numOfBlocksPerSMRegs;
    if (numOfBlocksPerSMThreads < minBlocksPerSM) minBlocksPerSM = numOfBlocksPerSMThreads;

    //the threadBlock amount is per SM multiply by number of SMs
    return minBlocksPerSM * currDeviceProperties.multiProcessorCount;
}

bool is_empty(int *c_id, int *p_id) {
    __sync_synchronize();
    return *c_id == *p_id;
}

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)rand_r(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

void rate_limit_wait(struct rate_limit_t *rate_limit) {
    while (!rate_limit_can_send(rate_limit)) {
        struct timespec t = {
            0,
            long(1. / (rate_limit->lambda * 1e-9) * 0.01)
        };
        nanosleep(&t, NULL);
    }
}

double distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    double distance_sqr = 0;
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

/* we won't load actual files. just fill the images with random bytes */
void load_images(uchar *images) {
    srand(0);
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        images[i] = rand() % 256;
    }
}

__device__ int arr_min(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int rhs, lhs;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            rhs = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            lhs = arr[tid];
            if (rhs != 0) {
                if (lhs == 0)
                    arr[tid] = rhs;
                else
                    arr[tid] = min(arr[tid], rhs);
            }
        }
        __syncthreads();
    }

    int ret = arr[arr_size - 1];
    return ret;
}

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__global__ void gpu_process_image(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}

__global__ void gpu_server(int* cpu2gpu_producer_idx_g, int* cpu2gpu_consumer_idx_g,uchar* cpu_gpu_queue_g, uchar* gpu_cpu_queue_g){
    __shared__ int histogram[PIXEL_VALUES];
    __shared__ int hist_min[PIXEL_VALUES];
    __shared__ uchar map[PIXEL_VALUES];
    __shared__ volatile uchar req;
    volatile uchar *image_in;
    int slot_size_bytes = 1 + SQR(IMG_DIMENSION);
    int tid = threadIdx.x;
    while (1) {
        if (tid == 0) {
            /* busy wait while there are no outstanding jobs or gpu_cpu_queue is full */
            int current_consume_slot = cpu2gpu_consumer_idx_g[blockIdx.x];
            while (cpu2gpu_producer_idx_g[blockIdx.x] == current_consume_slot){
                __threadfence_system();
            }
            uchar * tb_cpu_gpu_queue = cpu_gpu_queue_g + (blockIdx.x * Q_SLOTS  * slot_size_bytes);
            __threadfence_block();
            image_in = tb_cpu_gpu_queue + (current_consume_slot * slot_size_bytes);
            __threadfence_block();
            req = image_in[0];//check validator byte
            __threadfence_block();
            image_in++;//remove validator byte
            __threadfence_block();
        }
        __syncthreads();
        if (req != VALID){
            return;
        }

        if (tid < PIXEL_VALUES) {
            histogram[tid] = 0;
        }
        __threadfence_block();
        int current_consume_slot = cpu2gpu_consumer_idx_g[blockIdx.x];
        uchar * tb_cpu_gpu_queue = cpu_gpu_queue_g + (blockIdx.x * Q_SLOTS  * slot_size_bytes);
        image_in = tb_cpu_gpu_queue + (current_consume_slot * slot_size_bytes) + 1;

//        for (int i = tid; i < SQR(IMG_DIMENSION); i += gr.x) {
            atomicAdd(&histogram[image_in[tid]], 1);
//        }
        __threadfence_block();

        prefix_sum(histogram, PIXEL_VALUES);
        __threadfence_block();

        if (tid < PIXEL_VALUES) {
            hist_min[tid] = histogram[tid];
        }
        __threadfence_block();

        int cdf_min = arr_min(hist_min, PIXEL_VALUES);
        __threadfence_block();

        if (tid < PIXEL_VALUES) {
            int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
            map[tid] = (uchar)map_value;
        }

        __threadfence_block();
        uchar * tb_gpu_cpu_queue = gpu_cpu_queue_g + (blockIdx.x * Q_SLOTS  * SQR(IMG_DIMENSION));
        uchar *current_start_slot = tb_gpu_cpu_queue + (SQR(IMG_DIMENSION) * current_consume_slot);
        __threadfence_block();

//        for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
            current_start_slot[tid] =  map[image_in[tid]];
//        }
        __threadfence_system();

        if (tid == 0) {
            cpu2gpu_consumer_idx_g[blockIdx.x] = INCREASE_PC_POINTER(cpu2gpu_consumer_idx_g[blockIdx.x]);
            __threadfence_system();
        }
        __syncthreads();
    }
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}


enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};
int main(int argc, char *argv[]) {

    int mode = -1;
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    if (argc < 3) print_usage_and_die(argv[0]);

    if (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        threads_queue_mode = atoi(argv[2]);
        load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    uchar *images_in; /* we concatenate all images in one huge array */
    uchar *images_out;
    CUDA_CHECK( cudaHostAlloc(&images_in, NREQUESTS * SQR(IMG_DIMENSION), 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    load_images(images_in);
    double t_start, t_finish;

    /* using CPU */
    printf("\n=== CPU ===\n");
    t_start  = get_time_msec();
    for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx)
        process_image(&images_in[img_idx * SQR(IMG_DIMENSION)], &images_out[img_idx * SQR(IMG_DIMENSION)]);
    t_finish = get_time_msec();
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

    double total_distance = 0;

    /* using GPU task-serial.. just to verify the GPU code makes sense */
    printf("\n=== GPU Task Serial ===\n");

    uchar *images_out_from_gpu;
    CUDA_CHECK( cudaHostAlloc(&images_out_from_gpu, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    do {
        uchar *gpu_image_in, *gpu_image_out;
        CUDA_CHECK(cudaMalloc(&gpu_image_in, SQR(IMG_DIMENSION)));
        CUDA_CHECK(cudaMalloc(&gpu_image_out, SQR(IMG_DIMENSION)));

        t_start = get_time_msec();
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            CUDA_CHECK(cudaMemcpy(gpu_image_in, &images_in[img_idx * SQR(IMG_DIMENSION)], SQR(IMG_DIMENSION), cudaMemcpyHostToDevice));
            gpu_process_image<<<1, 1024>>>(gpu_image_in, gpu_image_out);
            CUDA_CHECK(cudaMemcpy(&images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)], gpu_image_out, SQR(IMG_DIMENSION), cudaMemcpyDeviceToHost));
        }
        total_distance += distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("distance from baseline %lf (should be zero)\n", total_distance);
        printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

        CUDA_CHECK(cudaFree(gpu_image_in));
        CUDA_CHECK(cudaFree(gpu_image_out));
    } while (0);

    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_start, 0, NREQUESTS * sizeof(double));

    double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_end, 0, NREQUESTS * sizeof(double));

    struct rate_limit_t rate_limit;
    rate_limit_init(&rate_limit, load, 0);

    CUDA_CHECK(cudaMemset(images_out_from_gpu, 0, NREQUESTS * SQR(IMG_DIMENSION)));

    const int IMAGE_SIZE = SQR(IMG_DIMENSION);

    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {
        //declare streams in and out buffers
        uchar *image_in_device_streams, *image_out_device_streams;
        /* allocating device memory for all number of streams * image size */
        CUDA_CHECK(cudaMalloc((void **)&image_in_device_streams, N_STREAMS * SQR(IMG_DIMENSION)));
        CUDA_CHECK(cudaMalloc((void **)&image_out_device_streams,N_STREAMS * SQR(IMG_DIMENSION)));

        /* initialize CUDA streams*/
        cudaStream_t streams[N_STREAMS];
        for (int i = 0; i < N_STREAMS; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        /* save per streams current handle request*/
        int request_per_stream[N_STREAMS];
        for(int s_i=0; s_i < N_STREAMS; s_i++){
            request_per_stream[s_i] = INVALID;
        }

        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            int chosen_stream = INVALID;
            /*finding stream to perform request */
            while (chosen_stream == INVALID){
                /* freeing stresms who finished */
                for (int stream_i = 0; stream_i < N_STREAMS; stream_i ++) {
                    if (request_per_stream[stream_i] != INVALID){// true when stream perform a request
                        if (cudaStreamQuery(streams[stream_i]) != cudaSuccess){//check if stream finished the job
                            continue;
                        }
                        req_t_end[request_per_stream[stream_i]] = get_time_msec();//record finished time
                        request_per_stream[stream_i] = INVALID;//mark stream job free
                    }
                    if (chosen_stream == INVALID){
                        chosen_stream = stream_i;//take stream to do next job
                    }
                }
            }
            if (!rate_limit_can_send(&rate_limit)) {
                --img_idx;
                continue;
            }
            req_t_start[img_idx] = get_time_msec();
            request_per_stream[chosen_stream] = img_idx;// recored stream current job

            CUDA_CHECK(cudaMemcpyAsync(image_in_device_streams + (chosen_stream * IMAGE_SIZE),
                                       images_in + (img_idx * IMAGE_SIZE), IMAGE_SIZE,
                                       cudaMemcpyHostToDevice, streams[chosen_stream]));
            gpu_process_image <<< 1, 1024, 0, streams[chosen_stream] >>> (image_in_device_streams + (chosen_stream * IMAGE_SIZE), image_out_device_streams + (chosen_stream * IMAGE_SIZE));
            CUDA_CHECK(cudaMemcpyAsync(images_out_from_gpu + (img_idx * IMAGE_SIZE),
                                       image_out_device_streams + (chosen_stream * IMAGE_SIZE),
                                       IMAGE_SIZE, cudaMemcpyDeviceToHost, streams[chosen_stream]));
        }

        CUDA_CHECK(cudaDeviceSynchronize());//wait all job to finished
        bool all_done = false;
        while (!all_done){
            all_done = true;
            /* freeing streams who finished */
            for (int stream_i = 0; stream_i < N_STREAMS; stream_i ++) {
                if (request_per_stream[stream_i] != INVALID){// true when stream perform a request
                    if (cudaStreamQuery(streams[stream_i]) != cudaSuccess){//check if stream finished the job
                        all_done = false;
                        continue;
                    }
                    req_t_end[request_per_stream[stream_i]] = get_time_msec();//record finished time
                    request_per_stream[stream_i] = INVALID;//mark stream job free
                }
            }
        }
        // cleanup streams environment
        for (int i = 0; i < N_STREAMS; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        CUDA_CHECK(cudaFree(image_out_device_streams));
        CUDA_CHECK(cudaFree(image_in_device_streams));
    }else if (mode == PROGRAM_MODE_QUEUE) {
//        int numberOfThreadBlocks = numOfThreadBlocksCalc(threads_queue_mode);
//         printf("Number of ThreadBlocks is %d\n", numberOfThreadBlocks);
//
//        // memory alloc
//        uchar *cpu_gpu_queue, *cpu_gpu_queue_g,*gpu_cpu_queue, *gpu_cpu_queue_g;
//
//        CUDA_CHECK(cudaHostAlloc(&cpu_gpu_queue, numberOfThreadBlocks * Q_SLOTS * (1 + image_size), 0));
//        CUDA_CHECK(cudaHostAlloc(&gpu_cpu_queue, numberOfThreadBlocks * Q_SLOTS * image_size, 0));
//
//        int *cpu2gpu_producer_idx, *cpu2gpu_consumer_idx;
//        int *cpu2gpu_producer_idx_g, *cpu2gpu_consumer_idx_g;
//
//        CUDA_CHECK(cudaHostAlloc(&cpu2gpu_producer_idx, numberOfThreadBlocks, 0));
//        CUDA_CHECK(cudaHostAlloc(&cpu2gpu_consumer_idx, numberOfThreadBlocks, 0));
//
//        CUDA_CHECK(cudaHostGetDevicePointer(&cpu2gpu_producer_idx_g, cpu2gpu_producer_idx, 0));
//        CUDA_CHECK(cudaHostGetDevicePointer(&cpu2gpu_consumer_idx_g, cpu2gpu_consumer_idx, 0));
//        CUDA_CHECK(cudaHostGetDevicePointer(&cpu_gpu_queue_g, cpu_gpu_queue, 0));
//        CUDA_CHECK(cudaHostGetDevicePointer(&gpu_cpu_queue_g, gpu_cpu_queue, 0));
//
//        int *req_by_tb = (int *) malloc(numberOfThreadBlocks * Q_SLOTS * sizeof(int));
//        int *last_slot_by_tb= (int *) malloc(numberOfThreadBlocks * sizeof(int));
//
//        //memsets
//        memset(cpu2gpu_producer_idx, 0, numberOfThreadBlocks * sizeof(int));
//        memset(cpu2gpu_consumer_idx, 0, numberOfThreadBlocks * sizeof(int));
//        memset(cpu_gpu_queue, 0, numberOfThreadBlocks * Q_SLOTS * (1 + image_size));
//        memset(gpu_cpu_queue, 0,  numberOfThreadBlocks * Q_SLOTS * image_size);
//        memset(last_slot_by_tb, 0, numberOfThreadBlocks * sizeof(int));
//        memset(req_by_tb, INVALID, numberOfThreadBlocks * Q_SLOTS * sizeof(int));
//
//        gpu_server <<< numberOfThreadBlocks, threads_queue_mode >>>
//                                              (cpu2gpu_producer_idx_g, cpu2gpu_consumer_idx_g, cpu_gpu_queue_g, gpu_cpu_queue_g);
//
//        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
//            printf("\ncurrent request: %d\n", img_idx);
//            int chosenThreadBlock = INVALID;
//            while (chosenThreadBlock == INVALID) {
//                for (int tb_it = 0; tb_it < numberOfThreadBlocks; tb_it++) {
//                    // read completed requests from tb
//                    while (!is_empty(&last_slot_by_tb[tb_it], &cpu2gpu_consumer_idx[tb_it])) {
//                        printf("tb %d queue is not empty\n", tb_it);
//                        int completeRequest = req_by_tb[(tb_it * Q_SLOTS) + last_slot_by_tb[tb_it]];
//                        printf("ThreadBlock id %d, request handled is %d, last_slot is %d, p_idx is %d, c_idx is %d\n",
//                               tb_it, completeRequest, last_slot_by_tb[tb_it], cpu2gpu_producer_idx[tb_it],
//                               cpu2gpu_consumer_idx[tb_it]);
//                        req_t_end[completeRequest] = get_time_msec();
//                        req_by_tb[(tb_it * Q_SLOTS) + last_slot_by_tb[tb_it]] = INVALID;
//                        dequeueJob(gpu_cpu_queue + (tb_it * Q_SLOTS), last_slot_by_tb[tb_it], completeRequest,
//                                   images_out_from_gpu, image_size);
//                        last_slot_by_tb[tb_it] = INCREASE_PC_POINTER(last_slot_by_tb[tb_it]);
//                    }
//                    if (chosenThreadBlock == INVALID &&
//                        !is_queue_full(cpu2gpu_producer_idx + tb_it, cpu2gpu_consumer_idx + tb_it)) {
//                        chosenThreadBlock = tb_it;
//                    }
//                }
//            }
//            if (!rate_limit_can_send(&rate_limit)) {
//                --img_idx;
//                continue;
//            }
//            req_t_start[img_idx] = get_time_msec();
//            req_by_tb[chosenThreadBlock * Q_SLOTS + cpu2gpu_producer_idx[chosenThreadBlock]] = img_idx;
//            printf("enqueue job %d to threadBlock %d p_id %d\n",img_idx, chosenThreadBlock,cpu2gpu_producer_idx[chosenThreadBlock]);
//            enqueueJob(cpu_gpu_queue + (chosenThreadBlock * Q_SLOTS * ((1 + image_size))), &cpu2gpu_producer_idx[chosenThreadBlock],
//                       img_idx, images_in, image_size, VALID);
//        }
//            /* wait until you have responses for all requests */
//            for (int tb_it = 0; tb_it < numberOfThreadBlocks; tb_it++) {
//                while (is_queue_full(cpu2gpu_producer_idx + tb_it, cpu2gpu_consumer_idx + tb_it)) {
//                }
//                req_by_tb[(tb_it * Q_SLOTS) + cpu2gpu_producer_idx[tb_it]] = INVALID;
//                enqueueJob(cpu_gpu_queue + (tb_it * Q_SLOTS * ((1 + image_size))),
//                        &cpu2gpu_producer_idx[tb_it],
//                        0, images_in, image_size, 0);
//                printf("added not valid request 0 to tb %d, p id %d c id %d l id %d\n", tb_it,
//                       cpu2gpu_producer_idx[tb_it], cpu2gpu_consumer_idx[tb_it], last_slot_by_tb[tb_it]);
//            }
//            CUDA_CHECK(cudaDeviceSynchronize());
//            printf("all devices finished!\n");
//            for (int tb_it = 0; tb_it < numberOfThreadBlocks; tb_it++) {
//                // read completed requests from tb
//                while (!is_empty(&last_slot_by_tb[tb_it], &cpu2gpu_consumer_idx[tb_it])) {
//                    int completeRequest = req_by_tb[(tb_it * Q_SLOTS) + last_slot_by_tb[tb_it]];
//                    printf("ThreadBlock id %d, request handled is %d, last_slot is %d, p_idx is %d, c_idx is %d\n",
//                           tb_it, completeRequest, last_slot_by_tb[tb_it], cpu2gpu_producer_idx[tb_it],
//                           cpu2gpu_consumer_idx[tb_it]);
//                    if (completeRequest != -1) {
//                        req_t_end[completeRequest] = get_time_msec();
//                        dequeueJob(gpu_cpu_queue + (tb_it * Q_SLOTS), last_slot_by_tb[tb_it], completeRequest,
//                                   images_out_from_gpu, image_size);
//                    }
//                    last_slot_by_tb[tb_it] = INCREASE_PC_POINTER(last_slot_by_tb[tb_it]);
//                }
//            }
        }
    else {
        assert(0);
    }
    double tf = get_time_msec();

    total_distance = distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (req_t_end[i] - req_t_start[i]);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("distance from baseline %lf (should be zero)\n", total_distance);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);
    return 0;
}
