@inline gpumod(x, n) = x + ((x < 1) - (x > n)) * n

@inline function gpucross(a, b)
  return (a[2] * b[3] - a[3] * b[2],
￼         a[3] * b[1] - a[1] * b[3],
￼         a[1] * b[2] - a[2] * b[1])
end

struct DFTMatrix{T} <: AbstractArray{T, 2}
  N::Int
  linearindices::LinearIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
  cartesianindices::CartesianIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
end
DFTMatrix(N) = DFTMatrix{Float32}(N, LinearIndices((N, N)), CartesianIndices((N, N)))
Base.size(dftm::DFTMatrix) = (dftm.N, dftm.N)

Base.getindex(dftm::DFTMatrix, i::Int) = getindex(dftm, dftm.cartesianindices[i])

forwardargument(i, j, dftm::DFTMatrix{T}) where T = - ((i-1) * (j-1)) / T(2)
backwardargument(i, j, dftm::DFTMatrix) = - forwardargument(i, j, dftm)

mygetindex(dftm::DFTMatrix, op::F, i::Int) where F = mygetindex(dftm, op, dftm.cartesianindices[i])
function mygetindex(dftm::DFTMatrix{T}, op::F, I)::Complex{T} where {T, F}
  i, j = I
#  @assert (1 <= i <= dftm.N) && (1 <= j <= dftm.N) "i = $i, j = $j, N = $N"
  (s, c) = sincospi(op(i, j, dftm))
  return c + im * s
end
function Base.getindex(dftm::DFTMatrix{T}, I::Vararg{Int, U})::Complex{T} where {T, U}
  return mygetindex(dftm, forwardargument, I)
end


function invmul!(y, dftm::DFTMatrix{T}, x::AbstractVector{U}) where {T, U}
  @threads for i in eachindex(y)
    yi = zero(promote_type(T, U))
    @inbounds for j in eachindex(x)
      yi += mygetindex(dftm, backwardargument, (i, j)) * x[j]
    end
    @inbounds y[i] = yi / length(y)
  end
  return y
end


struct OddEvenMatrix <: AbstractArray{Bool, 2}
  N::Int
  linearindices::LinearIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
  cartesianindices::CartesianIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
end
OddEvenMatrix(N) = OddEvenMatrix{Float32}(N, LinearIndices((N, N)), CartesianIndices((N, N)))
Base.size(oem::OddEvenMatrix) = (oem.N, oem.N)

Base.getindex(oem::OddEvenMatrix, i::Int) = getindex(oem, oem.cartesianindices[i])

function Base.getindex(oem::OddEvenMatrix, I::Vararg{Int, T})::Bool where {T}
  i, j = I
  return i <= (oem.N ÷ 2) ? j == 2 * (i-1) + 1 : j == 2 * (i - (oem.N ÷ 2))
end

struct FFTMatrix
  N::Int
  I2::Diagonal{Bool, Vector{Bool}}
  D2::SparseMatrixCSC{Complex{Int64}, Int64}
  R2::Matrix{Int}
  Ps::Vector{OddEvenMatrix}
end
function FFTMatrix(N)
  @assert ispow2(N)
  I2 = I(2)
  D2 = sparse([1 0; 0 -im])
  R2 = [1 1; 1 -1]
  return FFTMatrix(N, I2, D2, R2, [OddEvenMatrix(2^i) for i in 1:log2(N)])
end
Base.size(fftm::FFTMatrix) = (fftm.N, fftm.N)

function Base.getindex(fftm::FFTMatrix, I::Vararg{Int, T})::Bool where {T}
  i, j = I
  n = fftm.N
  while n > 2
    n = n ÷ 2
    ii = i ÷ 2
    jj = j ÷ 2
  end
end

#function Base.mul!(y, fftm::FFTMatrix, x)
#end
￼




