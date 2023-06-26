
struct FFTHelper{T, U}
  kx::Vector{Float64}
  ky::LinearAlgebra.Adjoint{Float64, Vector{Float64}}
  k²::Matrix{Float64}
  im_k⁻²::Matrix{ComplexF64}
  smoothingkernel::Matrix{ComplexF64}
  pfft!::T
  pifft!::U
end
function FFTHelper(NX, NY, Lx, Ly)
  kx = Vector(FFTW.fftfreq(NX, 2π / Lx * NX))
  ky = Vector(FFTW.fftfreq(NY, 2π / Ly * NY))'
  k² = (kx.^2 .+ ky.^2)
  im_k⁻² = -im ./ k²
  im_k⁻²[1, 1] = 0
  z = zeros(ComplexF64, NX, NY)
  kernelx = exp.(-(-min(6, NX÷2):min(6, NX÷2)).^2)
  kernely = exp.(-(-min(6, NY÷2):min(6, NY÷2)).^2)
  kernel2D = sqrt.(kernelx .* kernely')
  smoothingkernel = zeros(ComplexF64, NX, NY)
  for j in 1:min(NY, size(kernel2D, 2)), i in 1:min(NX, size(kernel2D, 1))
     smoothingkernel[i, j] = kernel2D[i, j]
  end
  smoothingkernel ./= sum(smoothingkernel)

  pfft! = plan_fft!(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  pifft! = plan_ifft!(z; flags=FFTW.ESTIMATE, timelimit=Inf)
  pfft! * smoothingkernel
  return FFTHelper(kx, ky, k², im_k⁻², smoothingkernel, pfft!, pifft!)
end


