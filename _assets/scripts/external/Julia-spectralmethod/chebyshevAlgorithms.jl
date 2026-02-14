#using Pkg
using LinearAlgebra

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




# Solves the eigenvalue problem u''''+Ru'''-su''=0 with vanishing Dirichlet
# and Neumann boundary conditions, using the modified Chebyshev Tau method.
function fourthOrderTestProblemOneSolver(N_int::Int,R::Float64)
    B = chebyshevSecondDerivativeMatrix(N_int+2,0) + R*chebyshevFirstDerivativeMatrix(N_int+2,0)
    #B = chebyshevSecondDerivativeMatrix(N_int,2) + R*chebyshevFirstDerivativeMatrix(N_int,2)
    Q = chebyshevSecondDerivativeMatrix(N_int,2)

    B1 = @view B[1:N_int-1,1:N_int+1]
    B2 = @view B[N_int:N_int+1,1:N_int+1]
    B4 = @view B[1:N_int-1,N_int+2:N_int+3]
    B5 = @view B[N_int:N_int+1,N_int+2:N_int+3]

    B3 = @view B[N_int+2:N_int+3,1:N_int+1]
    B6 = @view B[N_int+2:N_int+3,N_int+2:N_int+3]

    Q1 = @view Q[1:N_int-1,:]
    Q2 = @view Q[N_int:N_int+1,:]

    B5inv = inv(B5)
    
    M = B1*Q-B4*B5inv*B2*Q
    N = Q1 - B4*B5inv*Q2

    BC = zeros(4,N_int+3)
    BC[1, :] .= (-1).^(0:N_int+2)
    BC[2, :] .= 1
    BC[3, :] .= -(-1).^(0:N_int+2) .* (0:N_int+2).^2
    BC[4, :] .= (0:N_int+2).^2

    M1 = @view M[1:N_int-1,1:N_int-1]
    M2 = @view M[1:N_int-1,N_int:N_int+3]
    M3 = @view BC[1:4,1:N_int-1]
    M4 = @view BC[1:4,N_int:N_int+3]

    N1 = @view N[1:N_int-1,1:N_int-1]
    N2 = @view N[1:N_int-1,N_int:N_int+3]

    M4inv = inv(M4)

    values,vectors = eigen(M1-M2*M4inv*M3,N1-N2*M4inv*M3)

    vectors = [
    vectors;
    -M4inv*M3 *vectors
    ]

    # Normalize + sign
    vectors .= vectors ./ sum(abs, vectors; dims=1)
    vectors .*= reshape(sign.(vectors[1, :]), 1, :)
    

    # Tau values
    y = -B5inv*B2*Q*vectors + B5inv*Q2*vectors*diagm(values)
    tau = B3*Q*vectors + B6*y - y*diagm(values)
    tau = sum(abs, tau; dims=1)

    return values,vectors,tau
end


# Solves the eigenvalue problem u'''' - u''+Ru''=0 with eigenvalue
# located in the boundary condition:
# u(1) = u'(1) = 0
# u'(-1) = u''(-1) + h*u(-1) = 0
function fourthOrderTestProblemTwoSolver(N_int::Int,R::Float64)
    B = chebyshevSecondDerivativeMatrix(N_int+2,0)-I
    # Indicing mismatch sucks
    A = Matrix(1.0I, N_int+3, N_int+3)
    Q = chebyshevSecondDerivativeMatrix(N_int,2)

    B1 = @view B[1:N_int-1,1:N_int+1]
    B2 = @view B[N_int:N_int+1,1:N_int+1]
    B4 = @view B[1:N_int-1,N_int+2:N_int+3]
    B5 = @view B[N_int:N_int+1,N_int+2:N_int+3]

    B3 = @view B[N_int+2:N_int+3,1:N_int+1]
    B6 = @view B[N_int+2:N_int+3,N_int+2:N_int+3]

    A1 = @view A[1:N_int-1,:]
    A2 = @view A[N_int:N_int+1,:]
    A3 = @view A[N_int+2:N_int+3,:] 

    Q1 = @view Q[1:N_int-1,:]
    Q2 = @view Q[N_int:N_int+1,:]

    B5inv = inv(B5)

    M = B1*Q - B4*B5inv*(B2*Q-Q2+R*A2) - Q1 + R*A1

    BC1 = zeros(3,N_int+3)
    BC1[1, :] .= -(-1).^(0:N_int+2) .* (0:N_int+2).^2
    BC1[2, :] .= 1
    BC1[3, :] .= (0:N_int+2).^2

    BC2 = zeros(2,N_int+3)

    BC2[1, :] .=(-1).^(0:N_int+2) .*(0:N_int+2).^2 .*((0:N_int+2).^2 .- 1)
    BC2[2, :] .= -(-1).^(0:N_int+2)

    M = [M;BC1]
    N = zeros(size(M))


    values,vectors = eigen(vcat(M,BC2[1,:]'),vcat(N,BC2[2,:]'))

    mask = isfinite.(values)
    values = values[mask]
    vectors = vectors[:,mask]

    # Normalize + sign
    vectors .= vectors ./ sum(abs, vectors; dims=1)
    vectors .*= reshape(sign.(vectors[1, :]), 1, :)

    # Compute Tau values
    tau = (B3*Q - B6*B5inv*(B2*Q-Q2+R*A2) + R*A3)*vectors
    tau = sum(abs, tau; dims=1)

    return values,vectors,tau
end





