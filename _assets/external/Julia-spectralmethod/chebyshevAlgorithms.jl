using Pkg
using LinearAlgebra
#using SparseArrays




function chebyshevPolynomial(k::Int,x::Float64)

    if k==0; return 1; end
    if k==1; return x; end

    if k<70
        tmin0 = Float64
        tmin1 = x
        tmin2 = 1

        for j in 2:k
            tmin0 = 2*x*tmin1 - tmin2
            tmin2 = tmin1
            tmin1=tmin0
        end   

        return tmin0
 
    else return cos(k*acos(x))
    
    end

end

function chebyshevGaussLobattoNodesAndWeights(n::Int)
    x = -cos.((0:n)*pi/n)
    w = fill(pi/n,n+1)
    w[1] = w[1]/2
    w[end] = w[end]/2

    return x,w
end

function chebyshevVandermondeMatrix(n::Int)
    # Computes V_{ij} = T_j(x_i), where x_i are the Gauss-Lobatto nodes and weights.
    V = Array{Float64}(undef,(n+1,n+1))

    x = chebyshevGaussLobattoNodesAndWeights(n)[1]
    for i in 0:n, j in 0:n
        V[i+1,j+1] = chebyshevPolynomial(j,x[i+1])
    end
    return V
end

# Indicator function for nonzero entries for 2nd derivative.
function _chebyshevSecondDerivativeIndicator(i::Int,j::Int)
    if j >= i+2 && iseven(i+j+2)
        return 1
    else 
        return 0
    end
end

# General indicator function for nonzero entries for p'th derivative.
function _chebyshevPthDerivativeIndicator(p::Int,i::Int,j::Int)
    if j >= i + p && iseven(i+j+p)
        return 1
    else
        return 0
    end
end

# 
function _chebyshevSecondDerivativeEntries(i::Int,j::Int)
    return j*(j*j - i*i)
end

#
function _chebyshevThirdDerivativeEntries(i::Int,j::Int)
    return (j*((j^2)*(j^2 - 2) - 2*(j^2)*(i^2) + (i^2 -1)^2))/4
end

#
function _chebyshevFourthDerivativeEntries(i::Int,j::Int)
    return (j*(j^2 *(j^2 - 4)^2 - 3*j^4 * i^2 + 3*j^2 * i^4 - i^2 *(i^2 - 4)^2))/24
end

function chebyshevFirstDerivativeMatrix(N_int::Int, N_bou::Int)
    M = zeros(N_int + 1,N_int + N_bou + 1)

    for i in 0:N_int, j in 0:(N_int + N_bou)
        M[i+1,j+1] = _chebyshevPthDerivativeIndicator(1,i,j)*(2*j)
    end

    M[1,:] .= M[1,:]./2

    return M
end

#
function chebyshevSecondDerivativeMatrix(N_int::Int,N_bou::Int)
    M = zeros(N_int + 1,N_int + N_bou + 1)

    for i in 0:N_int, j in 0:(N_int + N_bou)
        M[i+1,j+1] = _chebyshevSecondDerivativeIndicator(i,j)*_chebyshevSecondDerivativeEntries(i,j)
    end

    M[1,:] .= M[1,:]./2

    return M
end

#
function chebyshevThirdDerivativeMatrix(N_int::Int,N_bou::Int)
    M = zeros(N_int + 1,N_int + N_bou + 1)

    for i in 0:N_int, j in 0:(N_int + N_bou)
        M[i+1,j+1] = _chebyshevPthDerivativeIndicator(3,i,j)*_chebyshevThirdDerivativeEntries(i,j)
    end

    M[1,:] .= M[1,:]./2

    return M
end

#
function chebyshevFourthDerivativeMatrix(N_int::Int,N_bou::Int)
    M = zeros(N_int + 1, N_int + N_bou + 1)

    for i in 0:N_int, j in 0:(N_int + N_bou)
        M[i+1,j+1] = _chebyshevPthDerivativeIndicator(4,i,j)*_chebyshevFourthDerivativeEntries(i,j)
    end

    M[1,:] .=M[1,:]./2

    return M
end






