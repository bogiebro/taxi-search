module Common
export neighborMin, neighborhoods, makePolicy

function tropicalDot(v::Array{Float64,1}, A::SparseMatrixCSC{Float64,Int})
  c = fill(Inf64, A.n)
  for vi in 1:A.n
    for ind in A.colptr[vi]:(A.colptr[vi+1]-1)
      nbr = A.rowval[ind]
      newval = v[nbr] + A.nzval[ind]
      if newval <= c[vi]
        c[vi] = newval
      end
    end
  end
  @assert ~any(isinf.(c))
  c
end

function neighborMin(A::SparseMatrixCSC{Float64,Int},v::Array{Float64,1})
    w = zeros(Int,A.m)
    c = fill(Inf64, A.m)
    for vi in 1:A.m
        for ind in A.colptr[vi]:(A.colptr[vi+1]-1)
            nbr = A.rowval[ind]
            newval =v[nbr]
            if newval <= c[vi]
                c[vi] = newval
                w[vi] = nbr
            end
        end
    end
    @assert ~any(isinf.(c))
    (c, w)
end

neighborhoods(g, n) = (g + speye(g))^n .> 0

function makePolicy(A, x)
  weighted = A .* sparse(1 ./ (x .+ 1))
  totals  = sum(weighted, 1)
  weighted .* sparse(1 ./ totals)
end

end
