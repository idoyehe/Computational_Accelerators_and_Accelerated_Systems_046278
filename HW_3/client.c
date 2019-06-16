#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <string.h>
#include <assert.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "common.h"

#define NREQUESTS 10000
#define INVALID -1
#define VALID 1

struct client_context {
    enum mode_enum mode;
    struct ib_info_t server_info;

    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_qp *qp;
    struct ibv_cq *cq;
    int server_key;
    long long server_addr;

    struct rpc_request* requests;
    struct ibv_mr *mr_requests;
    uchar *images_in;
    struct ibv_mr *mr_images_in;
    uchar *images_out;
    struct ibv_mr *mr_images_out;

    uchar *images_out_from_gpu;

    int producerIndex, consumerIndex;
    struct ibv_mr *mr_producerIndex;
    struct ibv_mr *mr_consumerIndex;

    int *requestPerTbSlot, *nextFetchedSlot;
};



bool is_queue_full(struct client_context *ctx, int *p_id, int *c_id) {
    struct ibv_send_wr wr;
    struct ibv_send_wr *bad_wr;
    struct ibv_wc wc;

    struct ibv_sge sg; /* scatter/gather element */

    /* RDMA send needs a gather element (local buffer)*/
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    sg.addr = (uintptr_t)&(ctx->producerIndex);
    sg.length = sizeof(int);
    sg.lkey = ctx->mr_producerIndex->lkey;

    wr.wr_id = 0;
    wr.sg_list = &sgl;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = p_id;
    wr.wr.rdma.rkey = ctx->server_info.rkey_producerIndex;

    if (ibv_post_send(ctx->qp, &er, &bad_wr)) {
        printf("ERROR: ibv_post_send() failed\n");
        exit(1);
    }

    struct ibv_wc wc;
    int ncqes;
    do {
        ncqes = ibv_poll_cq(ctx->cq, 1, &wc);
    } while (ncqes == 0);
    if (ncqes < 0) {
        printf("ERROR: ibv_poll_cq() failed\n");
        exit(1);
    }
    if (wc.status != IBV_WC_SUCCESS) {
        printf("ERROR: got CQE with error '%s' (%d) (line %d)\n", ibv_wc_status_str(wc.status), wc.status, __LINE__);
        exit(1);
    }

/* RDMA send needs a gather element (local buffer)*/
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    sg.addr = (uintptr_t)&(ctx->consumerIndex);
    sg.length = sizeof(int);
    sg.lkey = ctx->mr_consumerIndex->lkey;

    wr.wr_id = 0;
    wr.sg_list = &sgl;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = c_id;
    wr.wr.rdma.rkey = ctx->server_info.rkey_consumerIndex;

    if (ibv_post_send(ctx->qp, &er, &bad_wr)) {
        printf("ERROR: ibv_post_send() failed\n");
        exit(1);
    }

    do {
        ncqes = ibv_poll_cq(ctx->cq, 1, &wc);
    } while (ncqes == 0);
    if (ncqes < 0) {
        printf("ERROR: ibv_poll_cq() failed\n");
        exit(1);
    }
    if (wc.status != IBV_WC_SUCCESS) {
        printf("ERROR: got CQE with error '%s' (%d) (line %d)\n", ibv_wc_status_str(wc.status), wc.status, __LINE__);
        exit(1);
    }

    return ((ctx->producerIndex +1) % ctx->server_info.numberOfSlots) == ctx->consumerIndex;
}

bool is_empty(struct client_context *ctx,int *c_id, int *p_id) {
    struct ibv_send_wr wr;
    struct ibv_send_wr *bad_wr;
    struct ibv_wc wc;

    struct ibv_sge sg; /* scatter/gather element */

    /* RDMA send needs a gather element (local buffer)*/
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    sg.addr = (uintptr_t)&(ctx->consumerIndex);
    sg.length = sizeof(int);
    sg.lkey = ctx->mr_consumerIndex->lkey;

    wr.wr_id = 0;
    wr.sg_list = &sgl;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = p_id;
    wr.wr.rdma.rkey = ctx->server_info.rkey_consumerIndex;

    if (ibv_post_send(ctx->qp, &er, &bad_wr)) {
        printf("ERROR: ibv_post_send() failed\n");
        exit(1);
    }

    struct ibv_wc wc;
    int ncqes;
    do {
        ncqes = ibv_poll_cq(ctx->cq, 1, &wc);
    } while (ncqes == 0);
    if (ncqes < 0) {
        printf("ERROR: ibv_poll_cq() failed\n");
        exit(1);
    }
    if (wc.status != IBV_WC_SUCCESS) {
        printf("ERROR: got CQE with error '%s' (%d) (line %d)\n", ibv_wc_status_str(wc.status), wc.status, __LINE__);
        exit(1);
    }

    return (*c_id)  == ctx->consumerIndex;
}

void enqueueJob(struct client_context *ctx, uchar *threadBlockQueue, int *producerIndex, int image_idx,uchar * images_in, int image_size, uchar valid){
    struct ibv_send_wr wr;
    struct ibv_send_wr *bad_wr;
    struct ibv_wc wc;

    struct ibv_sge sg; /* scatter/gather element */

    /* RDMA send needs a gather element (local buffer)*/
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    sg.addr = (uintptr_t)&ctx->producerIndex;
    sg.length = sizeof(int);
    sg.lkey = ctx->mr_producerIndex->lkey;

    wr.wr_id = 0;
    wr.sg_list = &sgl;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = producerIndex;
    wr.wr.rdma.rkey = ctx->server_info.rkey_producerIndex;

    if (ibv_post_send(ctx->qp, &er, &bad_wr)) {
        printf("ERROR: ibv_post_send() failed\n");
        exit(1);
    }

    struct ibv_wc wc;
    int ncqes;
    do {
        ncqes = ibv_poll_cq(ctx->cq, 1, &wc);
    } while (ncqes == 0);
    if (ncqes < 0) {
        printf("ERROR: ibv_poll_cq() failed\n");
        exit(1);
    }
    if (wc.status != IBV_WC_SUCCESS) {
        printf("ERROR: got CQE with error '%s' (%d) (line %d)\n", ibv_wc_status_str(wc.status), wc.status, __LINE__);
        exit(1);
    }

    int offsetFromQueue = ctx->producerIndex * (1 + image_size);
    if(valid == VALID){
        memcpy(threadBlockQueue + offsetFromQueue +1, images_in + (image_idx * image_size), image_size);
    }
    memcpy(threadBlockQueue + offsetFromQueue, &valid, sizeof(uchar));
    __sync_synchronize();
    *producerIndex = INCREASE_PC_POINTER(*producerIndex);
    __sync_synchronize();
}

void dequeueJob(struct client_context *ctx, uchar *queue, int fetchedSlot, int image_idx, uchar *images_out, int image_size){
    int offsetFromQueue = (fetchedSlot * image_size);
    __sync_synchronize();
    memcpy(images_out + (image_idx * image_size), queue + offsetFromQueue, image_size);
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



void rpc_call(struct client_context *ctx,
              int request_id,
              void *img_in,
              void *img_out) {

    struct ibv_sge sg; /* scatter/gather element */
    struct ibv_send_wr wr; /* WQE */
    struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */
    struct ibv_wc wc; /* CQE */
    int ncqes; /* number of CQES polled */

    /* step 1: send request to server using Send operation */
    
    struct rpc_request *req = ctx->mr_requests->addr;
    req->request_id = request_id;
    req->input_rkey = img_in ? ctx->mr_images_in->rkey : 0;
    req->input_addr = (uintptr_t)img_in;
    req->input_length = SQR(IMG_DIMENSION);
    req->output_rkey = img_out ? ctx->mr_images_out->rkey : 0;
    req->output_addr = (uintptr_t)img_out;
    req->output_length = SQR(IMG_DIMENSION);

    /* RDMA send needs a gather element (local buffer)*/
    memset(&sg, 0, sizeof(struct ibv_sge));
    sg.addr = (uintptr_t)req;
    sg.length = sizeof(*req);
    sg.lkey = ctx->mr_requests->lkey;

    /* WQE */
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    wr.wr_id = request_id; /* helps identify the WQE */
    wr.sg_list = &sg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

    /* post the WQE to the HCA to execute it */
    if (ibv_post_send(ctx->qp, &wr, &bad_wr)) {
        printf("ERROR: ibv_post_send() failed\n");
        exit(1);
    }

    /* When WQE is completed we expect a CQE */
    /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
    /* The order between the two is not guarenteed */
    int got_send_cqe = 0,
	got_write_with_imm = img_out == NULL; /* One exception is the termination message which doesn't request an output */
    while (!got_send_cqe || !got_write_with_imm) {
        do {
            ncqes = ibv_poll_cq(ctx->cq, 1, &wc);
        } while (ncqes == 0);
        if (ncqes < 0) {
            printf("ERROR: ibv_poll_cq() failed\n");
            exit(1);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            printf("ERROR: got CQE with error '%s' (%d) (line %d)\n", ibv_wc_status_str(wc.status), wc.status, __LINE__);
            exit(1);
        }
        switch (wc.opcode) {
        case IBV_WC_SEND:
            got_send_cqe = 1;
            assert(wc.wr_id == request_id);
            break;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            got_write_with_imm = 1;
            assert(wc.imm_data == request_id);
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }
    }

    /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
    struct ibv_recv_wr recv_wr = {};
    if (ibv_post_recv(ctx->qp, &recv_wr, NULL)) {
        printf("ERROR: ibv_post_recv() failed\n");
        exit(1);
    }
}

void rpc_done(struct client_context *ctx) {
    rpc_call(ctx,
            -1, // Indicate termination
            NULL,
            NULL);
    printf("done\n");
}

struct client_context *setup_connection(int tcp_port) {
    /* first we'll connect to server via a TCP socket to exchange Infiniband parameters */
    int sfd;
    sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
        perror("socket");
        exit(1);
    }

    struct sockaddr_in server_addr;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); /* server is on same machine */
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(tcp_port);

    if (connect(sfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        perror("connect");
        exit(1);
    }

    /*ok let's start creating infiniband resources*/
    struct client_context *ctx = malloc(sizeof(struct client_context));

    /* get device list */
    struct ibv_device **device_list = ibv_get_device_list(NULL);
    if (!device_list) {
        printf("ERROR: ibv_get_device_list failed\n");
        exit(1);
    }

    /* select first (and only) device to work with */
    struct ibv_context *context = ibv_open_device(device_list[0]);

    /* create protection domain (PD) */
    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        printf("ERROR: ibv_alloc_pd() failed\n");
        exit(1);
    }

    /* create completion queue (CQ). We'll use same CQ for both send and receive parts of the QP */
    struct ibv_cq *cq = ibv_create_cq(context, 100, NULL, NULL, 0); /* create a CQ with place for 100 CQEs */
    if (!cq) {
        printf("ERROR: ibv_create_cq() failed\n");
        exit(1);
    }

    /* create QP */
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.qp_type = IBV_QPT_RC; /* we'll use RC transport service, which supports RDMA */
    qp_init_attr.cap.max_send_wr = 1; /* max of 1 WQE in-flight in SQ. that's enough for us */
    qp_init_attr.cap.max_recv_wr = OUTSTANDING_REQUESTS; /* max of 1 WQE's in-flight in RQ per outstanding request. that's more than enough for us */
    qp_init_attr.cap.max_send_sge = 1; /* 1 SGE in each send WQE */
    qp_init_attr.cap.max_recv_sge = 1; /* 1 SGE in each recv WQE */
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
        printf("ERROR: ibv_create_qp() failed\n");
        exit(1);
    }

    /* get server info (QP number and LID) */
    int ret;
    ret = recv(sfd, &ctx->server_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("recv");
        exit(1);
    }

    /* send our info to server */
    /*get our LID first */
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(context, IB_PORT_CLIENT, &port_attr);
    if (ret) {
        printf("ERROR: ibv_query_port() failed\n");
        exit(1);
    }
    struct ib_info_t my_info;
    my_info.lid = port_attr.lid;
    my_info.qpn = qp->qp_num;
    ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("send");
        exit(1);
    }
    /* we don't need TCP anymore. kill the socket */
    close(sfd);

    /* now need to connect the QP to the server''s QP.
     * this is a multi-phase process, moving the state machine of the QP step by step
     * until we are ready */
    struct ibv_qp_attr qp_attr;

    /*QP state: RESET -> INIT */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = IB_PORT_CLIENT;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ; /* Allow the server to read / write our memory through this QP */
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to INIT failed\n");
        exit(1);
    }


    /*QP: state: INIT -> RTR (Ready to Receive) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = ctx->server_info.qpn; /* qp number of server */
    qp_attr.rq_psn      = 0 ;
    qp_attr.max_dest_rd_atomic = 1; /* max in-flight RDMA reads */
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 0; /* No Network Layer (L3) */
    qp_attr.ah_attr.dlid = ctx->server_info.lid; /* LID (L2 Address) of server */
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = IB_PORT_CLIENT;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU| IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to RTR failed\n");
        exit(1);
    }

    /*QP: state: RTR -> RTS (Ready to Send) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.max_rd_atomic = 1;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to RTS failed\n");
        exit(1);
    }

    /* now let's populate the receive QP with recv WQEs */
    struct ibv_recv_wr recv_wr; /* this is the receive work request (the verb's representation for receive WQE) */
    int i;
    for (i = 0; i < OUTSTANDING_REQUESTS; i++) {
        memset(&recv_wr, 0, sizeof(struct ibv_recv_wr));
        recv_wr.wr_id = i;
        recv_wr.sg_list = NULL;
        recv_wr.num_sge = 0;
        if (ibv_post_recv(qp, &recv_wr, NULL)) {
            printf("ERROR: ibv_post_recv() failed\n");
            exit(1);
        }
    }

    ctx->context = context;
    ctx->pd = pd;
    ctx->qp = qp;
    ctx->cq = cq;

    return ctx;
}

void teardown_connection(struct client_context *ctx) {
    if (ctx->mode == MODE_RPC_SERVER) {
        ibv_dereg_mr(ctx->mr_images_in);
        ibv_dereg_mr(ctx->mr_images_out);
    }
    ibv_dereg_mr(ctx->mr_requests);
    ibv_dereg_mr(ctx->mr_producerIndex);
    ibv_dereg_mr(ctx->mr_consumerIndex);
    ibv_destroy_qp(ctx->qp);
    ibv_destroy_cq(ctx->cq);
    ibv_dealloc_pd(ctx->pd);
    ibv_close_device(ctx->context);
}

void allocate_and_register_memory(struct client_context *ctx)
{
    ctx->images_in = malloc(NREQUESTS * SQR(IMG_DIMENSION)); /* we concatenate all images in one huge array */
    ctx->images_out = malloc(NREQUESTS * SQR(IMG_DIMENSION));
    ctx->images_out_from_gpu = malloc(NREQUESTS * SQR(IMG_DIMENSION));
    struct rpc_request* requests = calloc(1, sizeof(struct rpc_request));

    assert(ctx->images_in && ctx->images_out && ctx->images_out_from_gpu);

    /* Register memory regions of our images input and output buffers for RDMA access */
    ctx->mr_images_in = ibv_reg_mr(ctx->pd, ctx->images_in, NREQUESTS * SQR(IMG_DIMENSION), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!ctx->mr_images_in) {
        perror("Unable to create input MR for RDMA");
        exit(1);
    }
    ctx->mr_images_out = ibv_reg_mr(ctx->pd, ctx->images_out_from_gpu, NREQUESTS * SQR(IMG_DIMENSION), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!ctx->mr_images_out) {
        perror("Unable to create output MR for RDMA");
        exit(1);
    }
    /* Register a memory region of our RPC request buffer */
    ctx->mr_requests = ibv_reg_mr(ctx->pd, requests, sizeof(struct rpc_request), 0);
    if (!ctx->mr_requests) {
        perror("Unable to create MR for sends");
        exit(1);
    }

    ctx->mr_requests = ibv_reg_mr(ctx->pd, requests, sizeof(struct rpc_request), 0);
    if (!ctx->mr_requests) {
        perror("Unable to create MR for sends");
        exit(1);
    }

    ctx->mr_producerIndex = ibv_reg_mr(ctx->pd,&ctx->producerIndex, sizeof(int), IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_producerIndex) {
        perror("Unable to create MR for producer index");
        exit(1);
    }

    ctx->mr_consumerIndex = ibv_reg_mr(ctx->pd,&ctx->consumerIndex, sizeof(int), IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_consumerIndex) {
        perror("Unable to create MR for consumer index");
        exit(1);
    }
}

void process_images(struct client_context *ctx)
{
    load_images(ctx->images_in);
    double t_start, t_finish;

    /* using CPU */
    printf("\n=== CPU ===\n");
    t_start  = get_time_msec();
    for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx)
        process_image(&ctx->images_in[img_idx * SQR(IMG_DIMENSION)], &ctx->images_out[img_idx * SQR(IMG_DIMENSION)]);
    t_finish = get_time_msec();
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

    /* using a GPU server */
    printf("\n=== GPU Server ===\n");

    double ti = get_time_msec();


    if (ctx->mode == MODE_RPC_SERVER) {
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            rpc_call(ctx, img_idx, 
                     &ctx->images_in[img_idx * SQR(IMG_DIMENSION)],
                     &ctx->images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)]);
        }
    } else {
        const int IMAGE_SIZE = SQR(IMG_DIMENSION);

        ctx->requestPerTbSlot = (int *) malloc(ctx->server_info.numberOfThreadBlocks * ctx->server_info.numberOfSlots * sizeof(int));
        ctx->nextFetchedSlot = (int *) malloc(ctx->server_info.numberOfThreadBlocks * sizeof(int));

        memset(ctx->nextFetchedSlot, 0, ctx->server_info.numberOfThreadBlocks * sizeof(int));
        memset(ctx->requestPerTbSlot, INVALID, ctx->server_info.numberOfThreadBlocks * ctx->server_info.numberOfSlots * sizeof(int));

        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            int chosenThreadBlock = INVALID;
            while (chosenThreadBlock == INVALID) {
                for (int threadBlock_i = 0; threadBlock_i < ctx->server_info.numberOfThreadBlocks; threadBlock_i++) {
                    // read completed requests from tb
                    while (!is_empty(ctx, ctx->nextFetchedSlot + threadBlock_i, ctx->server_info.consumerIndex + threadBlock_i)) {
                        int nextSlot = ctx->nextFetchedSlot[threadBlock_i];
                        int *currentRequestPerTbSlot = ctx->requestPerTbSlot + (threadBlock_i * ctx->server_info.numberOfSlots);
                        int completeRequest = currentRequestPerTbSlot[nextSlot];

                        dequeueJob(ctx->server_info.gpu2cpuQueue + (threadBlock_i * ctx->server_info.numberOfSlots * IMAGE_SIZE), nextSlot, completeRequest,
                                   ctx->images_out_from_gpu, IMAGE_SIZE);
                        currentRequestPerTbSlot[nextSlot] = INVALID;
                        ctx->nextFetchedSlot[threadBlock_i] = INCREASE_PC_POINTER(nextSlot);
                    }
                    if (chosenThreadBlock == INVALID &&
                        !is_queue_full(ctx, ctx->server_info.producerIndex + threadBlock_i, ctx->server_info.consumerIndex + threadBlock_i)) {
                        chosenThreadBlock = threadBlock_i;
                    }
                }
            }

            ctx->requestPerTbSlot[chosenThreadBlock * ctx->server_info.numberOfSlots +
                             ctx->server_info.producerIndex[chosenThreadBlock]] = img_idx;//save the request in the slot
            enqueueJob(ctx, ctx->server_info.cpu2gpuQueue + (chosenThreadBlock * ctx->server_info.numberOfSlots * ctx->server_info.slotSize2GPU),
                       ctx->server_info.producerIndex + chosenThreadBlock, img_idx, ctx->images_in, IMAGE_SIZE, VALID);
        }
        /* wait until you have responses for all requests and inform GPU to halt*/
        int all_done = false;
        while (!all_done) {
            all_done = true;
            for (int threadBlock_i = 0; threadBlock_i < ctx->numberOfThreadBlocks; threadBlock_i++) {
                // read completed requests from tb
                if (!is_empty(ctx->server_info.producerIndex + threadBlock_i, ctx->server_info.consumerIndex + threadBlock_i)) {
                    all_done = false;
                    continue;
                }
                while (!is_empty(ctx->nextFetchedSlot + threadBlock_i, ctx->server_info.consumerIndex + threadBlock_i)) {
                    int nextSlot = ctx->nextFetchedSlot[threadBlock_i];
                    int *currentRequestPerTbSlot = ctx->requestPerTbSlot + (threadBlock_i * ctx->server_info.numberOfSlots);
                    int completeRequest = currentRequestPerTbSlot[nextSlot];
                    dequeueJob(ctx->server_info.gpu2cpuQueue + (threadBlock_i * ctx->server_info.numberOfSlots * IMAGE_SIZE), nextSlot, completeRequest,
                               ctx->images_out_from_gpu, IMAGE_SIZE);
                    currentRequestPerTbSlot[nextSlot] = INVALID;
                    ctx->nextFetchedSlot[threadBlock_i] = INCREASE_PC_POINTER(nextSlot);
                }
            }
        }
        for (int threadBlock_i = 0; threadBlock_i < numberOfThreadBlocks; threadBlock_i++) {
            enqueueJob(ctx->server_info.cpu2gpuQueue + (threadBlock_i * ctx->server_info.numberOfSlots * ctx->server_info.slotSize2GPU), ctx->server_info.producerIndex + threadBlock_i, 0, 0, IMAGE_SIZE, 0);
        }

    double tf = get_time_msec();

    double total_distance = distance_sqr_between_image_arrays(ctx->images_out, ctx->images_out_from_gpu);
    printf("distance from baseline %lf (should be zero)\n", total_distance);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
}

int main(int argc, char *argv[]) {
    enum mode_enum mode;
    int tcp_port;

    parse_arguments(argc, argv, &mode, &tcp_port);
    if (!tcp_port) {
        printf("usage: %s <rpc|queue> <tcp port>\n", argv[0]);
        exit(1);
    }

    struct client_context *ctx = setup_connection(tcp_port);
    ctx->mode = mode;

    allocate_and_register_memory(ctx);

    process_images(ctx);

    rpc_done(ctx);

    teardown_connection(ctx);

    return 0;
}
