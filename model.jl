mutable struct Model
  W::Matrix{Float64}
  b::Vector{Float64}
end

subpart(a, s::Symbol) = getfield(a, s)
subpart(a::Real, s::Symbol) = a

Base.broadcast(f, args::Vararg{Union{Model,Real}}) = Model(
  f.((subpart(a, :W) for a in args)...), f.((subpart(a, :b) for a in args)...))
  
function Base.broadcast!(f, dst::Model, args::Vararg{Union{Model,Real}})
  dst.W .= f.((subpart(a, :W) for a in args)...)
  dst.b .= f.((subpart(a, :b) for a in args)...)
end

function predict(p, rho)
  linOut = p.W * rho + p.b
  max.(1, linOut), linOut
end

const lr=0.005

function getGrad(x, mask, rho, model)
  xbar, linOut = predict(model, rho)
  diff = xbar - x
  reluMask = (linOut .>= 1)
  Model(mask .* (diff * (rho .* reluMask)'), diff .* linOut)
end

function reinforce!(p, g, t)
  p .-= (lr * 0.999^t) .* g 
end
