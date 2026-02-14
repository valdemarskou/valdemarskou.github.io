include("external/Julia-spectralmethod/chebyshevAlgorithms.jl")
using Printf, CairoMakie


# Plotting eigenfunction
let 
    N = 50
    R = 150.0

    values,vectors,tau = fourthOrderTestProblemTwoSolver(N,R)
    xs = chebyshevGaussLobattoNodesAndWeights(N+2)[1]

    V = chebyshevVandermondeMatrix(N+2)
    ys = V*vectors[:,1]

    fig = Figure()
    ax = Axis(fig[1,1],
    xlabel="x",
    ylabel="u(x)",
    title="Eigenfunction, N = $(N), R = $(R)")

    scatterlines!(ax,xs,ys,color= :blue, linewidth=2,markercolor = :black,marker =:xcross,markersize = 7,
                    label = @sprintf("s ≈ %.5g, ||τ||₁ ≈ %.3g",values[1],tau[1]))

 
    axislegend(ax,"Eigenvalue and Tau magnitude",position=:rt, labelsize = 15)
    fig
    save(joinpath(@__DIR__, "output", "script3.svg"),fig)
end

