@views module Comparison
using Common
using Laplacians
using Distributions: wsample
export gridProblem, randPolicy, ptrPolicy, greedyPolicy,
  matrixPolicy, timeTilPickup, idleTime

# Problem generation
function gridProblem()
  g = grid2(10)
  n = 10^2
  p = rand(n) / 5
  M = rand(size(g)) + 1e-5
  M ./= sum(M,1)
  M -= diagm(diag(M))
  g, p, M
end

# Rho-independent search policies
randPolicy(A) = loc-> rand(nbrs(A, loc))

greedyPolicy(A,p) = loc-> begin
  neighbors = nbrs(A,loc)
  neighbors[indmax(p[neighbors])]
end

# CONTINUE HERE!

# Rho-dependent search policies
# We don't need Loc: just use rho!!!
# Should map rho to rho
# Just make a policy and multiply by it
modelPolicy(A,m) = (loc,rho)-> begin
  x = 1 ./ predict(m, rho)[1]
  x ./= sum(x)
  [wsample(A[:,l] .* x) for l in loc]
end



# Taxi simulations
# ADD THE RHO HERE!

function timeTilPickup(f, A, p, n)
  rho = randexp(A.m) * n
  notdone = fill(true, n)
  finishTimes = zeros(n)
  while any(notdone)
    finishTimes[notdone] .+= 1
    notdone .&= rand(n) .> p[loc]
    loc[notdone] .= f(loc[notdone])
  end
  finishTimes
end

function timeTilPickupIndep(pol, A, p, n)
  loc = rand(1:A.m, n)
  notdone = fill(true, n)
  finishTimes = zeros(n)
  while any(notdone)
    finishTimes[notdone] .+= 1
    notdone .&= rand(n) .> p[loc]
    loc[notdone] .= pol.(loc[notdone])
  end
  finishTimes
end

function idleTimeIndep(pol, A, p, M, n, r)
  loc = rand(1:A.m, n)
  rides = zeros(Int, n)
  idleTime = zeros(Int, n)
  while any(rides .< r)
    s = rand(n) .<= p[loc]
    rides[s] .+= 1
    loc[s] = wsample(M[:,loc], n)
    f = !s
    idleTime[f] .+= 1
    loc[f] .= pol.(loc[f])
  end
  idleTime
end

end
