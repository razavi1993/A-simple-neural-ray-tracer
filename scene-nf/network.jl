using Statistics
using Flux
using Plots
using Zygote
using StatsBase

const MIN_BIN_LEN = 0.025

function find_bins(x, w)
    x̂ = reshape(x, 1,:)
    return reshape(sum(x̂ .> w, dims=1), size(x))
end

function gather(x, bins, k)
    return getindex(x, bins .+ k.*(LinearIndices(bins) .- 1))
end

function coupling_transform(xa, xb, net, k)
    out = reshape(net(xa), 3k-1,:)
    n = size(out)[2]
    dₒᵤₜ = vcat(ones(1,n), softplus.(out[1:k-1,:]), ones(1,n))
    ωₒᵤₜ = vcat(zeros(1,n), MIN_BIN_LEN .+ (1.0 .- k.*MIN_BIN_LEN).*softmax(out[k:2k-1,:], dims=1))
    hₒᵤₜ = vcat(zeros(1,n), MIN_BIN_LEN .+ (1.0 .- k.*MIN_BIN_LEN).*softmax(out[2k:3k-1,:], dims=1))
    yₒᵤₜ = cumsum(hₒᵤₜ, dims=1)
    xₒᵤₜ = cumsum(ωₒᵤₜ, dims=1)
    bins = find_bins(xb, xₒᵤₜ)
    xₖ = gather(xₒᵤₜ[1:end-1,:], bins, k)
    xₖ₊₁ = gather(xₒᵤₜ[2:end,:], bins, k)
    δₖ = gather(dₒᵤₜ[1:end-1,:], bins, k)
    δₖ₊₁ = gather(dₒᵤₜ[2:end,:], bins, k)
    yₖ = gather(yₒᵤₜ[1:end-1,:], bins, k)
    yₖ₊₁ = gather(yₒᵤₜ[2:end,:], bins, k)
    ξ = (xb .- xₖ)./(xₖ₊₁ .- xₖ)
    sₖ = (yₖ₊₁ .- yₖ)./(xₖ₊₁ .- xₖ)
    ξ² = ξ.^2
    ξ₋₁² = (1.0 .- ξ).^2
    ξ₋₁ = ξ.*(1.0 .- ξ)
    ρ = sₖ .+ (δₖ₊₁ .+ δₖ .-2.0.*sₖ).*ξ₋₁
    αβ⁻¹ = yₖ .+ (yₖ₊₁ .- yₖ).*(sₖ.*ξ² .+ δₖ.*ξ₋₁)./ρ
    αβ⁻¹′ = prod(sₖ.^2 .*(δₖ₊₁.*ξ² .+2.0.*sₖ.*ξ₋₁ .+ δₖ.*ξ₋₁²)./(ρ.^2), dims=1)
    return αβ⁻¹, αβ⁻¹′
end

net1 = fmap(f64, Chain(Dense(10, 48, relu), Dense(48, 48, relu), Dense(48, 48, relu), Dense(48, 48, relu), Dense(48, 35)))
net2 = fmap(f64, Chain(Dense(10, 48, relu), Dense(48, 48, relu), Dense(48, 48, relu), Dense(48, 48, relu), Dense(48, 35)))

function normalizing_flow(x, nets, flipped, k)
    p₍ₓ₎ = ones(1,size(x)[2])
    for i=1:length(nets)
        if flipped[i] == true
            xa, xb = x[1:10,:], x[11:11,:]
            αβ⁻¹, αβ⁻¹′ = coupling_transform(xa, xb, nets[i], k)
            x = vcat(xa, αβ⁻¹)
            p₍ₓ₎ = p₍ₓ₎./αβ⁻¹′
        else
            xa, xb = x[1:1,:], x[2:11,:]
            αβ⁻¹, αβ⁻¹′ = coupling_transform(xb, xa, nets[i], k)
            x = vcat(αβ⁻¹, xb)
            p₍ₓ₎ = p₍ₓ₎./αβ⁻¹′
        end
    end
    return x, p₍ₓ₎
end

function loss(x,I)
    x, q₍ₓ₎ = nf(x[1:end-1,:])
    p₍ₓ₎ = x[end:end,:]./I
    return mean((p₍ₓ₎ .- q₍ₓ₎).^2 ./q₍ₓ₎)
end

ps = Flux.params(net1, net2)

nf(x) = normalizing_flow(x, [net1, net2], [true, false], 12)
