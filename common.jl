struct Info
  A::SparseMatrixCSC{Float64,Int}
  p::Vector{Float64}
  gammaM::SparseMatrixCSC{Float64,Int}
end

neighborhoods(g, n) = (g + speye(g))^n .> 0

function makePolicy(A, x)
  weighted = A .* sparse(1 ./ x)
  totals  = sum(weighted, 1)
  weighted .* sparse(1 ./ totals)
end

# Should add a thing for debugging that plots step sizes
const eps = 1e-6
function fixedPoint(f, initial)
  while true
    newVal = f(initial)
    if all(abs.(newVal .- initial) .< eps)
      return newVal
    end
    initial = newVal
  end
end

# Rho-independent search policies
matrixPolicy(P) = loc-> wsample(P[:, loc])
ptrPolicy(ptrs) = loc-> ptrs[loc]

