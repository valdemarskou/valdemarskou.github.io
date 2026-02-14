
include("external/Julia-spectralmethod/chebyshevAlgorithms.jl")
using Printf, CairoMakie

# plotting eigenfunctions
let 
    N = 60
    R = 0.0
    values,vectors,tau = fourthOrderTestProblemOneSolver(N,R)
    xs = chebyshevGaussLobattoNodesAndWeights(N + 2)[1]

    V = chebyshevVandermondeMatrix(N+2)
    ys1 = V*vectors[:,end]
    ys2 = V*vectors[:,end-1]
    ys3 = V*vectors[:,end-2]



    fig = Figure()
    ax = Axis(fig[1,1],
            xlabel="x",
            title="First three eigenfunctions, N = $(N), R = $(R)")

    scatterlines!(ax, xs, ys1, color= :blue, linewidth=2,markercolor = :black,marker =:xcross,markersize = 7,
                    label = @sprintf("s ≈ %.4g, ||τ||₁ ≈ %.3g",values[end],tau[end]))
    scatterlines!(ax, xs, ys2, color= :tomato, linewidth=2,markercolor = :black,marker =:xcross,markersize = 7,
                    label = @sprintf("s ≈ %.4g, ||τ||₁ ≈ %.3g",values[end-1],tau[end-1]))
    scatterlines!(ax, xs, ys3, color= :limegreen, linewidth=2,markercolor = :black,marker =:xcross,markersize = 7,
                    label = @sprintf("s ≈ %.4g, ||τ||₁ ≈ %.3g",values[end-2],tau[end-2]))


    axislegend(ax,"Eigenvalues and Tau magnitudes",position=:rb, labelsize = 15)
    fig
    save(joinpath(@__DIR__, "output", "script1.svg"),fig)
end



