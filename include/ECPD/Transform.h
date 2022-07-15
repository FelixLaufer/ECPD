#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#include "EigenTypes.h"

class Transform
{
 public:
  Transform(Quaternion q = Quaternion::Identity(), Vector3 p = Vector3::Zero())
    : pos(p)
	, quat(q)
  {}
  
  Transform(const Transform& other)
    : pos(other.pos)
	, quat(other.quat)
  {}
  
  Transform(const Vector& v) { fromVector(v); }

  inline Vector toVector() const
  {
    return (Vector(7) << pos, quat.w(), quat.x(), quat.y(), quat.z()).finished();
  }

  inline void fromVector(const Vector& v)
  {
    pos = v.segment<3>(0);
    quat = Quaternion(vec(0), vec(1), vec(2), vec(3)).normalized();
  }

  bool operator==(const Transform& other) const
  {
    return pos == other.pos && quat.w() == other.quat.w() && quat.vec() == other.quat.vec();
  }

  Vector3 pos;
  Quaternion quat;
};

#endif
