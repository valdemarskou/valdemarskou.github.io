using Pkg
include("chebyshevAlgorithms.jl")

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

    # Compute Tau Values
    y = -B5inv*B2*Q*vectors + B5inv*Q2*vectors*diagm(values)
    tau = B3*Q*vectors + B6*y - y*diagm(values)

    return values,vectors,tau
end


values,vectors,tau = fourthOrderTestProblemOneSolver(20,0.0)

size(values)
values[19]

tau[:,19]

function fourthOrderTestProblemTwoSolver(N_int::Int,lambda::Float64)
    B = chebyshevSecondDerivativeMatrix(N_int+2,0)-I
    #This indicing mismatch sucks
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

    M = B1*Q - B4*B5inv*(B2*Q-Q2+lambda*A2) - Q1 + lambda*A1

    BC = zeros(3,N_int+3)
    BC[1, :] .= (-1).^(0:N_int+2)
    BC[2, :] .= 1
    BC[3, :] .= -(-1).^(0:N_int+2) .* (0:N_int+2).^2
    BC[3, :] .= (0:N_int+2).^2

    M1 = @view M[:,1:N_int]
    M2 = @view M[:,N_int+1:N_int+3]
    M3 = @view BC[:,1:N_int]
    M4 = @view BC[:,N_int+1:N_int+3]

    M4inv = inv(M4)

    C = (-1).^(0:N_int+2) .*(0:N_int+2).^2 .*((0:N_int+2).^2 .- 1)
    D = -(-1).^(0:N_int+2) .* (0:N_int+2).^2

    N = M1 - M2*M4inv*M3

    N1 = @view N[:,1:N_int-1]
    N2 = @view N[:,N_int]

    (@view C[1])
    # Compute Tau values

    return N1
end


fourthOrderTestProblemTwoSolver(10,0.0)


