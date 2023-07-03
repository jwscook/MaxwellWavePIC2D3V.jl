using CommonSubexpressions

@inline function depositindicesfractions(s::AbstractShape, z::Float64, NZ::Int,
    NZ_Lz::Float64)
  zNZ = z * NZ_Lz # floating point position in units of cells
  # no need for unimod with offset arrays
  i = ceil(Int, zNZ) # cell number
  r = i - zNZ; # distance into cell i in units of cell width
  @assert 0 < r <= 1 "$z, $NZ, $NZ_Lz, $i"
  return gridinteractiontuple(s, i, r, NZ)
end

@inline gridinteractiontuple(::NGPWeighting, i, r, NZ) = ((i, 1), )
# no need for unimod with offset arrays
@inline function gridinteractiontuple(::AreaWeighting, i, r, NZ)
  return ((i, 1-r), (i+1, r))
end

#bsplineorder(::BSplineWeighting{N}) where N = N
#support(::BSplineWeighting{N}) where N = N+1
#Base.broadcastable(b::BSplineWeighting) = Ref(b)

bspline(::BSplineWeighting{@stat N}, x) where N = bspline(BSplineWeighting{Int(N)}(), x)

@inline bspline(::BSplineWeighting{0}, x) = ((1.0),)
@inline bspline(::BSplineWeighting{1}, x) = (x, 1-x)
function bspline(::BSplineWeighting{2}, x)
  @fastmath @cse begin
    a = 9/8 + 3/2*(x-1.5) + 1/2*(x-1.5)^2
    b = 3/4               -     (x-0.5)^2
    c = 9/8 - 3/2*(x+0.5) + 1/2*(x+0.5)^2
  end
  return (a, b, c)
end
function bspline(::BSplineWeighting{3}, x)
  @fastmath @cse begin
    a = 4/3 + 2*(x-2) + (x-2)^2 + 1/6*(x-2)^3
    b = 2/3           - (x-1)^2 - 1/2*(x-1)^3
    c = 2/3           - (x  )^2 + 1/2*(x  )^3
    d = 4/3 - 2*(x+1) + (x+1)^2 - 1/6*(x+1)^3
  end
  return (a, b, c, d)
end
function bspline(::BSplineWeighting{4}, x)
  @fastmath @cse begin
    a = 625/384 + 125/48*(x-2.5) + 25/16*(x-2.5)^2 + 5/12*(x-2.5)^3 + 1/24*(x-2.5)^4
    b = 55/96   -   5/24*(x-1.5) -   5/4*(x-1.5)^2 -  5/6*(x-1.5)^3 -  1/6*(x-1.5)^4
    c = 115/192                  -   5/8*(x-0.5)^2                  +  1/4*(x-0.5)^4
    d = 55/96   +   5/24*(x+0.5) -   5/4*(x+0.5)^2 +  5/6*(x+0.5)^3 -  1/6*(x+0.5)^4
    e = 625/384 - 125/48*(x+1.5) + 25/16*(x+1.5)^2 - 5/12*(x+1.5)^3 + 1/24*(x+1.5)^4
  end
  return (a, b, c, d, e)
end
function bspline(::BSplineWeighting{5}, x)
  @fastmath @cse begin
  a = 243/120 + 81/24*(x-3) + 9/4*(x-3)^2 + 3/4*(x-3)^3 + 1/8*(x-3)^4 + 1/120*(x-3)^5
  b = 17/40   -   5/8*(x-2) - 7/4*(x-2)^2 - 5/4*(x-2)^3 - 3/8*(x-2)^4 -  1/24*(x-2)^5
  c = 22/40                 - 1/2*(x-1)^2               + 1/4*(x-1)^4 +  1/12*(x-1)^5
  d = 22/40                 - 1/2*(x+0)^2               + 1/4*(x-0)^4 -  1/12*(x-0)^5
  e = 17/40   +   5/8*(x+1) - 7/4*(x+1)^2 + 5/4*(x+1)^3 - 3/8*(x+1)^4 +  1/24*(x+1)^5
  f = 243/120 - 81/24*(x+2) + 9/4*(x+2)^2 - 3/4*(x+2)^3 + 1/8*(x+2)^4 - 1/120*(x+2)^5
  end
  return (a, b, c, d, e, f)
end

@inline indices(::BSplineWeighting{N}, i) where N = (i-fld(N, 2)):(i+cld(N, 2))

for N in 0:2:10
  @eval _bsplineinputs(::BSplineWeighting{@stat $(N+1)}, i, centre) = (i, 1 - centre)
  @eval function _bsplineinputs(::BSplineWeighting{@stat $N}, i, centre)
    q = centre > 0.5
    return (i + q, q + 0.5 - centre)
  end
end

@inline function gridinteractiontuple(s::BSplineWeighting{N}, i, centre::T, NZ
    ) where {N,T}
#  (j, z) = if isodd(N)
#    (i, 1 - centre)
#  else
#    q = centre > 0.5
#    (i + q, q + 0.5 - centre)
#  end
  j, z = _bsplineinputs(s, i, centre)
  inds = indices(s, j)
  fractions = bspline(s, z)
  #@assert sum(fractions) ≈ 1 "$(sum(fractions)), $fractions"
  return zip(inds, fractions)
end


function deposit!(z::AbstractArray{<:Number, 2}, s::AbstractShape, x, y, NX_Lx, NY_Ly, w::Number)
  NX, NY = size(z)
  for (j, wy) in depositindicesfractions(s, y, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, x, NX, NX_Lx)
      z[i,j] += wx * wy * w
    end
  end
end

function deposit!(z::AbstractArray{<:Number, 3}, s::AbstractShape, x, y, NX_Lx, NY_Ly, w1, w2, w3)
  NV, NX, NY = size(z)
  @assert NV == 3
  for (j, wy) in depositindicesfractions(s, y, NY, NY_Ly)
    for (i, wx) in depositindicesfractions(s, x, NX, NX_Lx)
      wxy = wx * wy
      @muladd z[1,i,j] = z[1,i,j] + wxy * w1
      @muladd z[2,i,j] = z[2,i,j] + wxy * w2
      @muladd z[3,i,j] = z[3,i,j] + wxy * w3
    end
  end
end


