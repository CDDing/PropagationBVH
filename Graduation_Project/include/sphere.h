#ifndef SPHEREH
#define SPHEREH

#include <hittable.h>

class sphere : public hittable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r, material* m,bool movable) : movable(movable),center(cen), radius(r), mat_ptr(m) {
		vec3 rvec = vec3(radius, radius, radius);
		bbox = aabb(cen - rvec, cen + rvec);
	};
	#define RND (curand_uniform(&local_rand_state))
	__device__ void changePosition(curandState* global_state) override {
		if (movable) {
			curandState local_rand_state = *global_state;
			vec3 offset = vec3((RND - 0.5) * 0.01 , (RND - 0.5) * 0.01, (RND - 0.5) * 0.01);
			center = center + offset;
			bbox = bbox + offset;
		}
	}
	__device__ bool hit(const ray& r, float maxt, hit_record& rec) const {
		vec3 oc = r.origin() - center;
		float a = dot(r.direction(), r.direction());
		float b = dot(oc, r.direction());
		float c = dot(oc, oc) - radius * radius;
		float discriminant = b * b - a * c;
		if (discriminant > 0) {
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < maxt && temp > 0.001f) {
				rec.t = temp;
				rec.p = r.at(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat = mat_ptr;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < maxt && temp > 0.001f) {
				rec.t = temp;
				rec.p = r.at(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat = mat_ptr;
				return true;
			}
		}
		return false;
	}
	__device__ aabb bounding_box() const override { return bbox; }

	__device__ bool isLeaf() const override { return true; }

private:
	vec3 center;
	float radius;
	material* mat_ptr;
	aabb bbox;
	bool movable;
};

#endif