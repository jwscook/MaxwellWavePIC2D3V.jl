struct ElectrostaticBoris
  t::SVector{3, Float64}
  t²::Float64
  dt_2::Float64
end
function ElectrostaticBoris(B::AbstractVector, dt::Float64)
  t = (@SArray [B[1], B[2], B[3]]) * dt / 2
  t² = dot(t, t)
  return ElectrostaticBoris(t, t², dt / 2)
end
function (boris::ElectrostaticBoris)(vx, vy, vz, Ex, Ey, q_m)
  v² = vx^2 + vy^2 + vz^2
  @assert v² <= 1 "v² = $(v²), vx = $vx, vy = $vy, vz = $vz"
  γ = 1 / sqrt(1 - v²)
  Ē₂ = (@SArray [Ex, Ey, 0.0]) * boris.dt_2 * q_m
  vγ⁻ = (@SArray [vx, vy, vz]) * γ + Ē₂
  vγ⁺ = vγ⁻ + cross(vγ⁻ + cross(vγ⁻, boris.t), boris.t) * q_m^2 * 2 / (1 + q_m^2 * boris.t²)
  vγ  = vγ⁺ + Ē₂
  @inbounds vγ² = vγ[1]^2 + vγ[2]^2 + vγ[3]^2
  γ = sqrt(1 + vγ²)
  return vγ / γ
end

struct ElectromagneticBoris
  dt_2::Float64
  ElectromagneticBoris(dt::Float64) = new(dt / 2)
end

function (boris::ElectromagneticBoris)(vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, q_m)
  v² = vx^2 + vy^2 + vz^2
  @assert v² <= 1 "v² = $(v²), vx = $vx, vy = $vy, vz = $vz"
  γ = 1 / sqrt(1 - v²)
  θ = boris.dt_2 * q_m
  t = (@SArray [Bx, By, Bz]) * θ
  tscale = 2 / (1 + dot(t, t))
  Ē₂ = (@SArray [Ex, Ey, Ez]) * θ
  vγ⁻ = (@SArray [vx, vy, vz]) * γ + Ē₂
  vγ⁺ = vγ⁻ + cross(vγ⁻ + cross(vγ⁻, t), t) * tscale
  vγ  = vγ⁺ + Ē₂
  @inbounds vγ² = vγ[1]^2 + vγ[2]^2 + vγ[3]^2
  γ = sqrt(1 + vγ²)
  return vγ / γ
end

