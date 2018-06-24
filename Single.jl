module Multi
using Common
using Laplacians
using Params
using PlotLog

function walkUnit(A, rho, lambda, model)
  x = predict(model, rho)[1]
  failPolicy = makePolicy(A, x)
  failRho = failPolicy * rho
  mins = neighborMin(predict(model, failRho)[1], A)
  p = lambda ./ (rho .+ 1)
  s = p
  f = (1 - p) .* mins
  s + f
end

function summarize(model, g, logs)
  report(logs, :gradW, [minimum(g.W), mean(g.W), maximum(g.W)])
  report(logs, :gradB, [minimum(g.b), mean(g.b), maximum(g.b)])
  report(logs, :w, [minimum(model.W), mean(model.W), maximum(model.W)])
  report(logs, :b, [minimum(model.b), mean(model.b), maximum(model.b)])
end

function fleetIterWalk(A, lambda, lograd)
  logs = MakePlot([:gradW, :gradB, :w, :b])
  mask = neighborhoods(A, lograd)'
  nodes = size(lambda)[1]
  model = Model(zeros(nodes,nodes), rand(nodes) * 10)
  for i in 1:100
    rho = round.(randexp(length(lambda)))
    for j in 1:20
      x = walkUnit(A, rho, lambda, model)
      grad = getGrad(x, mask, rho, model)
      summarize(model, grad, logs)
      train(model, grad, i)
    end
  end
  model, showlog(logs)
end

function simpleTest()
  seed = rand(1:1000)
  print("Seed: "); println(seed)
  srand(seed)
  A = grid2(5)
  lambda = rand(A.m) / 2
  fleetIterWalk(A, lambda, 3)
end

end
