﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <curand_kernel.h>
#include <math_functions.h>
#include <hittable_list.h>
#include <ray.h>
#include <sphere.h>
#include <hittable_list.h>
#include <material.h>
#include <vec3.h>
#include <camera.h>
#include <bvh.h>
#include <triangle.h>
#include <obj.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


unsigned char* dbackground_image;
hittable_list** world;
bvh_node** bvh_list;
bool prop = false;
camera** cam;
int object_counts = 1000;
curandState* random_state;
int** test;
// convert floating point rgb color to 8-bit integer
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }
__device__ int rgbToInt(float r, float g, float b) {
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}
__device__ int vectorgb(vec3 color) {
	return rgbToInt(color.x() * 255, color.y() * 255, color.z() * 255);
}
__global__ void movCam(camera** ca, int direction, int weight) {
	(*ca)->moveorigin(direction, weight);
}
__global__ void RotateCam(camera** ca, vec3 direction) {
	(*ca)->rotate(direction);
}
__global__ void ManipulateVFOV(camera** ca, int x) {
	(*ca)->changevfov(x);
}
extern "C" void moveCamera(int direction, int weight) {
	movCam << <1, 1 >> > (cam, direction, weight);
}
extern "C" void RotateCamera(int x, int y) {
	RotateCam << <1, 1 >> > (cam, vec3(x, y, 0));
}
extern "C" void manivfov(int x) {
	ManipulateVFOV << <1, 1 >> > (cam, x);
}

__global__ void AddSphere(hittable_list** world,int cnt) {
	hittable** list = (*world)->get_list();
	bvh_node* bvh = (bvh_node*)list[0];
	curandState local_rand_state;
	curand_init(cnt, 0, 0, &local_rand_state);
	printf("추가");
	bvh->add(new sphere(vec3(RND*10, RND*10, RND*10), 4, new lambertian(vec3(RND, RND, RND)),0));
}

int ballcnt = 0;
extern "C" void GenerateSphere() {
	AddSphere << <1, 1 >> > (world,ballcnt++);
}

__global__ void CalculatePerPixel(hittable_list** world, camera** camera, curandState* global_rand_state, unsigned int* g_odata, int imgh, int imgw) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int i = blockIdx.x * bw + tx;
	int j = blockIdx.y * bh + ty;
	int index = i + j * imgh;

	curandState local_rand_state = global_rand_state[index];
	vec3 color(0, 0, 0);

	int depth = (*camera)->max_depth;
	int spp = (*camera)->samples_per_pixel;
	float rate = 1 / float(spp);
	ray r = (*camera)->get_ray(&local_rand_state, i, j);
	for (int i = 0; i < spp; i++) {
		color += (*camera)->ray_color(&local_rand_state, r, depth, world);
	}
	color *= rate;
	global_rand_state[index] = local_rand_state;
	g_odata[i + j * imgw] = vectorgb(color);
}
__global__ void initCamera(camera** ca,unsigned char* background_image,int iw,int ih) {
	*ca = new camera(16.0 / 9.0, //종횡비
		1600,                    //이미지 가로길이
		5,                       //픽셀당 샘플수
		5,                      //반사 횟수
		90,                      //시야각
		vec3(-20, 0, 0),         //카메라 위치 
		vec3(0, 0, -1),          //바라보는곳
		vec3(0, 1, 0),           //업벡터
		vec3(0.5f, 0.7f, 1));      //배경색
	(*ca)->Setbackground(background_image,iw,ih);
}
__global__ void initWorld(hittable_list** world, int object_counts) {
	(*world) = new hittable_list(object_counts);
}
#define RND (curand_uniform(&local_rand_state))
__global__ void addObjects(curandState* global_state, hittable_list** world, int object_counts) {
	curand_init(0, 0, 0, &global_state[0]);
	curandState local_rand_state = *global_state;
	(*world)->add(new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)),0));

	//(*world)->add(new triangle(vec3(50, 50, 50), vec3(-50, 50, 50), vec3(50, -50, 50), new metal(vec3(0.5, 0.7, 0.8),0)));

	(*world)->add(new sphere(vec3(0, 200, 0), 100, new light(vec3(1, 1, 1)),0));
	int sphere_count = 10;




	for (int a = -sphere_count; a < sphere_count; a++) {
		for (int b = -sphere_count; b < sphere_count; b++) {
			
			float choose_mat = RND;
			vec3 center(a + RND, 0.2, b + RND);
			if (choose_mat < 0.8f) {
				(*world)->add(new sphere(center, 0.2, new lambertian(vec3(RND * RND, RND * RND, RND * RND)),1));
			}
			else if (choose_mat <= 1.0f) {
				(*world)->add(new sphere(center, 0.2, new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.0f/*0.5f * RND*/),1));
			}
			else {
				(*world)->add(new sphere(center, 0.2, new dielectric(1.5),1));
			}
		}
	}
}
__global__ void makeBVH(curandState* global_state, hittable_list** world, bvh_node** bvh_list, int object_counts) {
	printf("%d개\n", (*world)->now_size);
	curand_init(0, 0, 0, &global_state[0]);
	curandState local_rand_state = *global_state;
	(*world) = new hittable_list((hittable*)new bvh_node(world, bvh_list, &local_rand_state), object_counts);

}
__global__ void addTriangle(hittable_list** world, vec3 a, vec3 b, vec3 c, vec3 color) {
	(*world)->add(new triangle(a, b, c, new dielectric(2.0f)));

}
__global__ void Random_Init(curandState* global_state, int ih) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;
	unsigned int pixel_index = x + y * ih;
	curand_init(pixel_index, 0, 0, &global_state[pixel_index]);
}

__global__ void initMesh(hittable_list** tmp, int obj_counts) {
	(*tmp) = new hittable_list(obj_counts);
}
__global__ void mergeMesh(hittable_list** world, hittable_list** tmp, bvh_node** node, curandState* rand_state) {
	curand_init(0, 0, 0, &rand_state[0]);
	curandState local_rand_state = *rand_state;

	(*world)->add(new bvh_node(tmp, node, &local_rand_state));
}

void ReadOBJ(const char* objlist[], int obj_counts, const vec3 translist[], const vec3 scalelist[]) {
	Assimp::Importer importer;
	for (int c = 0; c < obj_counts; c++) {
		char str[100] = "resource/";
		strcat(str, objlist[c]);
		printf("%s\n", str);
		const aiScene* scene = importer.ReadFile(str, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
		{
			printf("Read File Exception\n");
		}
		vec3 translate = translist[c];
		vec3 scale = scalelist[c];
		int cnt = 0;
		for (int i = 0; i < scene->mNumMeshes; i++) {
			auto mesh = scene->mMeshes[i];
			curandState* mesh_state;
			cudaMalloc(&mesh_state, sizeof(curandState));

			bvh_node** node;
			hittable_list** tmp;
			cudaMalloc(&tmp, sizeof(hittable_list*));
			int startIdx = 1 << 30;
			while (true) {
				if ((startIdx >> 1) > mesh->mNumFaces) { startIdx >>= 1; }
				else { break; }
			}

			cudaMalloc((void**)&node, startIdx * sizeof(bvh_node*));

			initMesh << <1, 1 >> > (tmp, startIdx);

			for (int j = 0; j < mesh->mNumFaces; j++) {
				auto Face = mesh->mFaces[j];
				vec3 a(mesh->mVertices[Face.mIndices[0]].x, mesh->mVertices[Face.mIndices[0]].y, mesh->mVertices[Face.mIndices[0]].z);
				vec3 b(mesh->mVertices[Face.mIndices[1]].x, mesh->mVertices[Face.mIndices[1]].y, mesh->mVertices[Face.mIndices[1]].z);
				vec3 c(mesh->mVertices[Face.mIndices[2]].x, mesh->mVertices[Face.mIndices[2]].y, mesh->mVertices[Face.mIndices[2]].z);
				a *= scale;				b *= scale;				c *= scale;
				a += translate;		b += translate;		c += translate;
				aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
				aiColor4D diffuse, specular, ambient;
				aiGetMaterialColor(material, AI_MATKEY_COLOR_AMBIENT, &ambient);
				aiGetMaterialColor(material, AI_MATKEY_COLOR_SPECULAR, &specular);
				aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diffuse);
				aiColor4D sum = diffuse + specular + ambient;
				vec3 color(sum.r, sum.g, sum.b);
				//color = vec3(1.0f, 0.0f, 0.0f);
				addTriangle << <1, 1 >> > (tmp, a, b, c, color);

			}
			cudaDeviceSynchronize();
			mergeMesh << <1, 1 >> > (world, tmp, node, mesh_state);
		}

	}
}

extern "C" void initCuda(dim3 grid, dim3 block, int image_height, int image_width, int pixels) {
	//cudaDeviceSetLimit(cudaLimitStackSize, 256 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
	cudaMalloc(&random_state, pixels * sizeof(curandState));
	Random_Init << <grid, block, 0 >> > (random_state, image_height);

	const int bytes_per_pixel = 3;
	//배경 이미지 읽기
	auto n = bytes_per_pixel;
	int iw, ih;
	auto background_image = stbi_load("resource/background.png", &iw, &ih, &n, bytes_per_pixel);
	if (background_image == nullptr) {
		printf("이미지 로딩 에러\n");
	}
	cudaMalloc(&dbackground_image, iw * ih * bytes_per_pixel);
	cudaMemcpy(dbackground_image, background_image, iw * ih * bytes_per_pixel, cudaMemcpyHostToDevice);

	//랜덤 초기화
	cudaMalloc((void**)&world, sizeof(hittable*));
	initWorld << <1, 1 >> > (world, object_counts); cudaDeviceSynchronize();

	//월드 초기화 OBJ 읽기 및 카메라 등
	const char* objlist[] = { "buff-doge.obj" };      //읽을 OBJ 리스트, 및의 배열들과 순서 맞춰야함
	const vec3 translist[] = {
										vec3(10.0f,10.0f,0.0f) };  //위에서 읽을 OBJ를 옮겨주는 벡터
	const vec3 scalelist[] = {
										vec3(5.0f,5.0f,5.0f) };   //위에서 읽을 OBJ의 크기를 바꿔주는 벡터
	//ReadOBJ(objlist, 1, translist, scalelist);

	//여기까지 OBJ 읽기
	curandState* objectinit;
	cudaMalloc(&objectinit, sizeof(curandState));
	addObjects << <1, 1 >> > (objectinit, world, object_counts);
	cudaMalloc(&cam, sizeof(camera*));
	initCamera << <1, 1 >> > (cam,dbackground_image,iw,ih);

	cudaDeviceSynchronize();        //쿠다커널이 종료될때까지 기다리는 함수. 위의 world에 오브젝트 다 담길때까지 기다림.
	//BVH 생성 중 오브젝트 담기는 것 방지용

	curandState* bvh_state;
	cudaMalloc(&bvh_state, sizeof(curandState));
	cudaMalloc((void**)&bvh_list, object_counts * sizeof(bvh_node*));
	makeBVH << <1, 1 >> > (bvh_state, world, bvh_list, object_counts);

}
__global__ void printTraversalCount(hittable_list** world) {
	auto bvh = (bvh_node*)(*world)->get_list()[0];
	printf("평균 순회 횟수 : %f\n",bvh->getTraversal());
}
extern "C" void generatePixel(dim3 grid, dim3 block, int sbytes,
	unsigned int* g_odata, int imgh, int imgw) {
	//Propagation<<<1,1>>>(world,random_state,prop);
	CalculatePerPixel << <grid, block, sbytes >> > (world, cam, random_state, g_odata, imgh, imgw);
	//printTraversalCount << <1, 1 >> > (world);
}