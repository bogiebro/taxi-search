module NewPlot

using RecipesBase
using Plots
using TaxiFlow

@recipe function plot(t::TaxiFlow.TaxiRoute)
    n = length(t.aj)
    ai=1:n
    arx = [t.x[ai]';t.x[t.aj]';NaN*ones(n)']
    ary = [t.y[ai]';t.y[t.aj]';NaN*ones(n)']
    linewidth --> 1
    axis := nothing
    legend := nothing
    size --> (1000,1000)

    (gai,gaj,av) = findnz(t.g)
    garx = [t.x[gai]';t.x[gaj]';NaN*ones(length(gai))']
    gary = [t.y[gai]';t.y[gaj]';NaN*ones(length(gai))']

    @series begin
      seriesalpha := 0.2
      arrow := 1.1
      seriescolor := :black
      garx[:],gary[:]
    end

    @series begin
      arrow := 1.1
      seriescolor := :blue
      arx[:],ary[:]
    end

    @series begin
      markersize := 2
      mc := t.p
      seriestype := :scatter
      t.x, t.y
    end
end

function plotRoute()
  r = TaxiFlow.makeRoute()
  plot(r)
end

end
