mutable struct Path
    points::Array{Float64}
    is_diffuse::Array{Bool}
    f_values::Array{Float64}
end

mutable struct Memory
    points::Array{Float64}
    f_product::Array{Float64}
end

function calculate_product(f_values::Array{Float64}, is_diffuse::Array{Bool})
    y = reverse(cumprod(reverse(f_values[1:1,2:end]), dims=2))[is_diffuse]'
    return y
end

function sample_points(memory::Memory, n_samples::Int)
    sampled_inds = sample(1:length(memory.f_product), n_samples)
    points = memory.points[:,sampled_inds]
    f_product = memory.f_product[:,sampled_inds]
    return points, f_product
end

function add_remove_path(memory::Memory, path::Path, max_memory_size::Int)
    if length(memory.f_product) < max_memory_size
       m = minimum([max_memory_size - length(memory.f_product), length(path.f_values)])
       memory.f_product = hcat(memory.f_product, path.f_values[1:1,1:m])
       memory.points = hcat(memory.points, path.points[1:11,1:m])
   else
       m = length(path.f_values)
       memory.f_product = hcat(memory.f_product[1:1,m+1:end], path.f_values)
       memory.points = hcat(memory.points[1:11,m+1:end], path.points)
   end
end
