# local coordinate
function local_(n::vec)
    w = unit(n)
    if abs(w.x) > 0.9
        a = vec(0, 1, 0)
    else
        a = vec(1, 0, 0)
    end
    v = unit(cross(w, a))
    u = cross(w, v)
    return u, v, w
end

# uniform sampling
function sample_direction(nl::vec, s::Float64, r::Float64)
    u, v, w = local_(nl)
    x = cos(2*π*s)*sqrt(1 - r*r)
    y = sin(2*π*s)*sqrt(1 - r*r)
    z = r
    return x*u + y*v + z*w
end

# radiance function
function radiance(r::ray, world::hitable, depth::Int, memory::Memory, max_memory_size::Int, max_training_steps::Int, training_step::Int, nf)
    put = vec(1,1,1)
    points = zeros(11,0)
    is_diffuse = zeros(Bool,1,0)
    f_values = zeros(1,0)
    for i=1:depth
        hit_surface = hit(world, r, 0.0001, typemax(Float64))
        if !ismissing(hit_surface)
            if typeof(hit_surface.mater) == lambertian
                n = hit_surface.normal
                color = hit_surface.mater.color
                point = hit_surface.point
                x = vcat(rand(1,1), vec_to_arr(point), vec_to_arr(n), vec_to_arr(color), rand(1,1))
                points = hcat(points, x)
                is_diffuse = hcat(is_diffuse, true)
                u, pᵤ = nf(x)
                brdf = color/(1.0*π)
                ω = sample_direction(n, u[1,1], u[11,1])
                cosθ = dot(ω, n)
                pdf = pᵤ[1]/(2.0*π)
                f_values = hcat(f_values, reshape([maxe(brdf*cosθ/pdf)], 1,1))
                put = put*brdf*cosθ/pdf
                r = ray(point, ω)
            elseif typeof(hit_surface.mater) == mirror
                n = hit_surface.normal
                ω = r.d - 2.0*dot(r.d,n)*n
                is_diffuse = hcat(is_diffuse, false)
                put = put*hit_surface.mater.ref
                f_values = hcat(f_values, reshape([maxe(hit_surface.mater.ref)], 1,1))
                r = ray(hit_surface.point, ω)
            else
                if i > 1 && training_step <= max_training_steps
                    f_values = hcat(f_values, reshape([maxe(hit_surface.mater.emit)], 1,1))
                    path = Path(points, is_diffuse, calculate_product(f_values, is_diffuse))
                    if training_step < max_training_steps
                        add_remove_path(memory, path, max_memory_size)
                    end
                end
                return put*hit_surface.mater.emit
            end
        else
            if i > 1 && training_step <= max_training_steps
                f_values = hcat(f_values, 0.)
                path = Path(points, is_diffuse, calculate_product(f_values, is_diffuse))
                if training_step < max_training_steps
                    add_remove_path(memory, path, max_memory_size)
                end
            end
            return vec(0,0,0)
        end
    end
    return vec(0,0,0)
end
