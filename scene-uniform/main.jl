using FileIO, ImageView, Plots

include("vector.jl")
include("camera.jl")
include("hitable.jl")
include("surfaces.jl")
include("radiance.jl")

# render
function render(world::hitable, cam::camera, nx=300, ny=300, ns=100, depth=10)
    res = zeros(ny, nx, 3)
    for s in 1:ns
        for j in 1:ny
            for i in 1:nx
                r = ray(cam, (i+rand())/nx, (j+rand())/ny)
                cl = radiance(r, world, depth)
                res[ny-j+1, nx-i+1, :] += [cl.x, cl.y, cl.z]
            end
        end
    end
    return sqrt.(clamp.(res./ns))
end

# scene
function scene()
    surfaces = []
    push!(surfaces, yz_rect(0, 0, 1, 0, 1, vec(1,0,0), lambertian(vec(0.75,0.25,0.25))))
    push!(surfaces, xz_rect(0.999, 0.25, 0.75, 0.25, 0.75, vec(0,1,0), light(vec(5,5,5))))
    push!(surfaces, xz_rect(0, 0, 1, 0, 1, vec(0,1,0), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, yz_rect(1, 0, 1, 0, 1, vec(-1,0,0), lambertian(vec(0.25,0.25,0.75))))
    push!(surfaces, xz_rect(1, 0, 1, 0, 1, vec(0,-1,0), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, xy_rect(1, 0, 1, 0, 1, vec(0,0,-1), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, sphere(0.15, vec(0.3,0.151,0.55), mirror(vec(0.98,0.98,0.98))))
    push!(surfaces, sphere(0.15, vec(0.7,0.151,0.35), lambertian(vec(0.25,0.75,0.25))))
    return hit_list(surfaces)
end

nx = 350
ny = 350
ns = 32

cam = camera(vec(0.5,0.5,-1.3), vec(0.5,0.5,0), vec(0,1,0), 40.0, nx/ny)
img = render(scene(), cam, nx, ny, ns)
imshow(img)
save("C:\\Users\\javan\\Desktop\\scene-uniform\\img.png", img)
