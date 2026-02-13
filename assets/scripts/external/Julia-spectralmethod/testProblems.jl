#using Pkg
using CairoMakie
using ComplexPlots
using Printf
include("chebyshevAlgorithms.jl")



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
end

# plotting (complex) eigenvalues
let 
    N = 99 #N < 100
    R = 5.0
    M = 40 #M << N/2 for proper convergence

    values = fourthOrderTestProblemOneSolver(N,R)[1][end-(M-1):end]
    fig = Figure()
    ax = Axis(fig[1,1],
            xlabel="Re",
            ylabel="Im",
            title="First M = $(M) eigenvalues")
    scatter!(ax,values,marker=:xcross,markersize=8,color=:blue,label="Eigenvalues")
    #fig = sphereplot(values,markersize=5)

    #axislegend(ax,"Eigenvalues",position=:rt, labelsize = 15)
    fig
end


# plotting convergence of ||τ||₁ for first M eigenvalues
let 
    Nmax = 50
    M = 3
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
end



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
end


# plotting convergence of ||τ||₁ for the eigenvalue
let 
    Nmax = 50
    R = 0.0
    xs = 1:5:Nmax
    value = fourthOrderTestProblemTwoSolver(Nmax,R)[1][1]
    tau=[]

    fig = Figure()
    ax = Axis(fig[1,1],
            xlabel="N",
            ylabel="||τ||₁",
            yscale = log10,
            title="Tau magnitudes for increasing resolution")
    
    for j in xs
        tau = [tau;(fourthOrderTestProblemTwoSolver(j,R)[3][1])]
    end

    # Filter zero values
    mask = (xs .> 0) .& (tau .> 0)
    scatter!(ax,xs[mask],tau[mask],marker=:xcross,
                    label = @sprintf("s ≈ %.3g",value))
    
    
    axislegend(ax,"Eigenvalue",position=:rt, labelsize = 15)
    fig

end
