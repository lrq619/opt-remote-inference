import ctp

def forward_case_study(metrics):
    inference_latencys, comm_overheads, inter_tensor_sizes = metrics
    with ctp.append_run("forward_case_study") as run:
        run.monitor("inference_latencys", inference_latencys)
        run.monitor("comm_overheads", comm_overheads)
        run.monitor("inter_tensor_sizes", inter_tensor_sizes)


run = ctp.append_run("generate_case_study")
def generate_case_study(metrics):
    inference_latencys, comm_overheads, inter_tensor_sizes = metrics
    run.collect("inference_latencys", sum(inference_latencys))
    run.collect("comm_overheads", sum(comm_overheads))
    run.collect("inter_tensor_sizes", sum(inter_tensor_sizes))
    run.stop_collect()