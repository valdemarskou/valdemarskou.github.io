include("external/Julia-spectralmethod/chebyshevAlgorithms.jl")
using Printf, CairoMakie

# plotting convergence of ||τ||₁ for first M eigenvalues
let 
    Nmax = 60
    M = 5
    R = 1.0
    xs = 10:5:Nmax

    values = fourthOrderTestProblemOneSolver(Nmax,R)[1]

    fig = Figure()
    ax = Axis(fig[1,1],
            xlabel="N",
            ylabel="||τ||₁",
            yscale = log10,
            title="Tau magnitudes for increasing resolution, first $(M) eigenvalues")

    for i in 0:(M-1)
        tau = []

        for j in xs
            tau = [tau;(fourthOrderTestProblemOneSolver(j,R)[3])[end-i]]
        end

        scatter!(ax,xs,tau,marker=:xcross,
                    label = @sprintf("s ≈ %.3g",values[end-i]))
    end

    axislegend(ax,"Eigenvalues",position=:rt, labelsize = 15)
    fig
    save(joinpath(@__DIR__, "output", "script2.svg"),fig)
end
