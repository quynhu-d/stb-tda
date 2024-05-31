using Ripserer
using LinearAlgebra
using Plots
using Distances

using Printf
function get_representatives(data, k, metric=Distances.euclidean)
    diagram_cocycles = ripserer(data; reps=true, metric=metric)
    k_ = min(size(diagram_cocycles[2])[1], k)
    most_persistent_co = diagram_cocycles[2][end-k_+1:end]
    filtration = diagram_cocycles[2].filtration
    cycles = [reconstruct_cycle(filtration, mpc) for mpc in most_persistent_co]
    @printf("%d %d\n", k_, length(data))
    return diagram_cocycles, cycles
end

using NPZ
function process_data(
        data_path="semantic_space\\ru_cbow_word_center_100.npy", dir_path="holes\\RU\\words\\", chunk_num="1112", 
        metric=Distances.cosine_dist
)
    vars = npzread(data_path);
    data = [vars[i,:] for i in 1:size(vars,1)];
    @printf("%d\n", size(data)[1])
    
    diagram, cycles = get_representatives(data, 500, metric);
    display(plot(diagram))
    npzwrite("$(dir_path)h0_$(chunk_num).npy", stack([stack([bd[1], bd[2]], dims=1) for bd in diagram[1]]));
    npzwrite("$(dir_path)h1_$(chunk_num).npy", stack([stack([bd[1], bd[2]], dims=1) for bd in diagram[2]]));
    
    scycles = [stack([vertices(sx) for sx in cycle], dims=1) for cycle in cycles];
    sscycles = []
    for sc in scycles
        sscycles = [sscycles; sc]
    end
    sscycles = Matrix{Int64}(sscycles);
    npzwrite("$(dir_path)ru_word_holes_$(chunk_num).npy", sscycles)
    return diagram, cycles
end