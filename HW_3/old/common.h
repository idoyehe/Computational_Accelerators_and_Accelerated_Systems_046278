#pragma once

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#define IMG_DIMENSION 32
#define OUTSTANDING_REQUESTS 100

#define SQR(a) ((a) * (a))

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char uchar;

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

struct rpc_request
{
    int request_id; /* Returned to the client via RDMA write immediate value; use -1 to terminate */

    /* Input buffer */
    int input_rkey;
    int input_length;
    uint64_t input_addr;

    /* Output buffer */
    int output_rkey;
    int output_length;
    uint64_t output_addr;
};

#define IB_PORT_SERVER 1
#define IB_PORT_CLIENT 2

/////////////////////////////////////////////////////////////////////////////////////////////////////

struct ib_info_t {
    int lid;
    int qpn;

    uchar *cpu2gpuQueue, *gpu2cpuQueue;
    int *producerIndex, *consumerIndex;

    uint32_t rkey_cpu2gpuQueue, rkey_gpu2cpuQueue;
    uint32_t rkey_producerIndex, rkey_consumerIndex;

    int numberOfSlots;
    int numberOfThreadBlocks;
    int slotSize2GPU, slotSize2CPU;
    /* TODO add additional server rkeys / addresses here if needed */
};

enum mode_enum {
    MODE_RPC_SERVER,
    MODE_QUEUE,
};

void parse_arguments(int argc, char **argv, enum mode_enum *mode, int *tcp_port);

#ifdef __cplusplus
}
#endif