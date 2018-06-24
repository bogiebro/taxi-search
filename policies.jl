module ProportionalAgent

walkMulti(info) = oldX-> x-> info.p .* (1 + info.gammaM * oldX) .+
  (1 .- info.p) .* (makePolicy(info.A, x).' * x)

walk(info) = x-> info.p .+ (1 .- info.p) .* (makePolicy(info.A,x).' * x)

policy(info) = matrixPolicy(makePolicy(info.A, fixedPoint(walk(info), 1 ./ p)))

function policyMulti(info)
  w = walkMulti(info)
  xs = fixedPoint(x-> fixedPoint(w(x), x), zeros(info.A.m))
  matrixPolicy(makePolicy(info.A, xs))
end

end


module GreedyAgent

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

walk(info) = x-> info.p .+ (1 - info.p) .* (1 + neighborMin(info.A,x)[1])

walkMulti(info) = oldX-> x-> info.p .* (1 .+ info.gammaM * oldX) .+
  (1 - info.p) .* (1 + neighborMin(info.A,x)[1])

policy(info) = ptrPolicy(neighborMin(fixedPoint(walk(info), 1 ./ p))[2])

function policyMulti(info)
  w = walkMulti(info)
  xs = fixedPoint(x-> fixedPoint(w(x), x), zeros(info.A.m))
  ptrPolicy(neighborMin(xs)[2])
end

end

module Fleet

walk(info) = (model, rho)-> begin
  x = predict(model, rho)[1]
  failPolicy = makePolicy(info.A, x)
  failRho = failPolicy * rho
  failX = failPolicy.' * predict(model, failRho)[1]
  p = info.p ./ (rho .+ 1)
  f = (1 .- p) .* failX
  p .+ f
end

#=
walkMulti(info) = oldModel-> (model, rho)-> begin
  x = predict(model, rho)[1]
  failPolicy = makePolicy(info.A, x)
  failRho = failPolicy * rho
  failX = failPolicy.' * predict(model, failRho)[1]
  p = info.p ./ (rho .+ 1)
  # We don't know what rho will be once we're done with our trip
  # It will be different based on which M we take. Can't use linearity
  # in the same way. 
  # We could always assume that taxis will be in the stationary distribution
  # of the policy. That might generalize better. Think about this.
  oldX = predict(oldModel, stationaryRho)[1]
  p .* (1 .+ info.gammaM * oldX) .+ (1 .- p) .* failX
end
=#

function policy(info)
  m = Model(zeros(nodes,nodes), rand(nodes) * 10)
  train(walk(info), m, ()->round.(randexp(length(info.p))))
  modelPolicy(m)
end

end

