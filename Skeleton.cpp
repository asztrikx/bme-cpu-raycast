#include "framework.h"

#define dbg if(false)

float floatEqual(float subject, float number) {
	float eps = 0.001;
	return subject > number - eps && subject < number + eps;
}

vec3 perpendicular(vec3 v) {
	float coos[] = {v.x, v.y, v.z};
	float result[3];

	int zeroCount = 0;
	float sum = 0.0f;
	int lastNonZeroIndex = 0;
	for (int i = 0; i < 3; i++) {
		if (coos[i] == 0) {
			zeroCount++;
		} else {
			sum += coos[i];
			lastNonZeroIndex = i;
		}
		result[i] = 1;
	}

	sum -= coos[lastNonZeroIndex];
	result[lastNonZeroIndex] = -sum / coos[lastNonZeroIndex];
	return vec3(result[0], result[1], result[2]);
}

// second is smaller
std::pair<float,float> quardratic(float a, float b, float c) {
	float discr = b*b-4.0f*a*c;
	if (discr < 0) return std::make_pair(nanf(""),nanf(""));
	float sqrtDiscr = sqrtf(discr);
	float t1 = (-b+sqrtDiscr)/2.0f/a;
	float t2 = (-b-sqrtDiscr)/2.0f/a;
	return std::make_pair(t1,t2);
}

float rayQuadratic(float a, float b, float c) {
	std::pair<float,float> r = quardratic(a,b,c);
	float t1 = r.first;
	float t2 = r.second;
	if (isnan(t1)) return t1;
	if (t1 <= 0) return nanf("");
	return (t2>0) ? t2 : t1;
}

float rnd() { return (float)rand() / RAND_MAX; }
float const eps = 0.0001f;

void print(std::string text, vec3 v) {
	printf("%s: %ef %ef %ef\n", text.c_str(), v.x, v.y, v.z);
}

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 kd, vec3 ks, float shininess): ka(kd*M_PI), kd(kd), ks(ks), shininess(shininess) {}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 start, vec3 dir): start(start), dir(normalize(dir)) {}
};

class Intersectable {
  protected:
	Material* material;

	Intersectable(Material* material): material(material) {}
  public:
	virtual Hit intersect(Ray const& ray) = 0;
};

class Sphere: public Intersectable {
	vec3 center;
	float radius;
  public:
    Sphere(vec3 center, float radius, Material* material)
	: Intersectable(material), center(center), radius(radius) {}

	Hit intersect(Ray const& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(ray.dir, dist) * 2.0f;
		float c = dot(dist, dist) - radius * radius;

		float sol = rayQuadratic(a,b,c);
		if (isnan(sol)) return hit;

		hit.t = sol;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;
		hit.material = material;
		return hit;
	}
};

struct Plane: public Intersectable {
	vec3 normal;
	vec3 point;
	
	Plane(vec3 normal, vec3 point, Material* material)
	: Intersectable(material), normal(normalize(normal)), point(point) {}

	Hit intersect(Ray const& ray) {
		Hit hit;
		float divisor = dot(ray.dir, normal);
		if (floatEqual(divisor, 0.0f)) return hit;
		float t = dot(point - ray.start, normal) / divisor;
		if (t < 0) return hit;
		hit.position = ray.start + t*ray.dir;
		hit.normal = normal;
		if (dot(-ray.dir, hit.normal) < 0) { hit.normal = -hit.normal; }
		hit.material = material;
		hit.t = t;
		return hit;
	}
};

class Cylinder: public Intersectable {
	vec3 start;
	float height;
	vec3 dir;
	float r;
	vec3 normal;

	std::vector<Plane> planes;

  public:
	Cylinder(vec3 start, float height, float r, vec3 dir, Material* material)
	: Intersectable(material), start(start), height(height), dir(normalize(dir)), r(r) {
		normal = perpendicular(dir);
		planes.push_back(Plane(dir, start + dir * height, material));
		planes.push_back(Plane(dir, start, material));
	}

	Hit intersect(Ray const& ray) {
		Hit hit;
		Hit hitPlane;

		for (Plane plane: planes) {
			//dbg print("ray.start", ray.start);
			//dbg print("ray.dir", ray.dir);
			Hit hitPlaneCandidate = plane.intersect(ray);
			// did hit
			if (hitPlaneCandidate.t > 0 && length(hitPlaneCandidate.position - plane.point) <= r) {
				dbg printf("%ef %ef\n", hitPlaneCandidate.position.z, hitPlaneCandidate.t);
				if (hitPlane.t < 0 || hitPlaneCandidate.t < hitPlane.t) {
					if(hitPlane.t >= 0 && hitPlaneCandidate.t < hitPlane.t) dbg printf("xD");
					hitPlane = hitPlaneCandidate;
				}
			}
		}

		vec3 A = ray.dir - dir*dot(ray.dir,dir);
		vec3 B = ray.start - start - dir*dot(ray.start,dir)+dir*dot(start,dir);

		float a = dot(A,A);
		float b = 2*dot(A,B);
		float c = dot(B,B)-r*r;
		std::pair<float,float> sols = quardratic(a,b,c);
		if (isnan(sols.first)) return hitPlane;

		float t = sols.second;
		vec3 position = ray.start + ray.dir * t;
		float distFromStart = dot(dir, position - start);
		if (t < 0 || distFromStart > height || distFromStart < 0) {
			t = sols.first;
			position = ray.start + ray.dir * t;
			distFromStart = dot(dir, position - start);
			if (t < 0 || distFromStart > height || distFromStart < 0) return hitPlane;
		}
		if (hitPlane.t > 0 && hitPlane.t < t) return hitPlane;

		vec3 center = start + distFromStart*dir;

		hit.t = t;
		hit.material = material;
		hit.normal = (position - center) / r;
		hit.position = position;
		return hit; //min of hit hitplane
	}
};

class Paraboloid: public Intersectable {
	vec3 start, dir;
	float height, radius;
	vec3 eNormal, eP;
	vec3 f;

 public:
	Paraboloid(vec3 start, vec3 dir, float height, float radius, Material* material)
	: Intersectable(material), start(start), dir(normalize(dir)), height(height), radius(radius) {
		eNormal = dir;
		vec3 T = start+height*dir+radius*perpendicular(dir);

		float a = dot(dir, dir) - pow(dot(eNormal,dir),2);
		float b = 2*dot(dir,start-T) + 2*dot(eNormal,start-T)*dot(eNormal,dir);
		float c = dot(start-T,start-T) - powf(dot(eNormal,start-T),2);

		float x;
		if (floatEqual(a, 0.0f)) {
			x = -c/b;
		} else {
			auto sols = quardratic(a,b,c);
			printf("sols: %ef %ef\n-", sols.first,sols.second);
			x = sols.first;
			if (x <= 0) printf("fak\n");
		}

		f = start+x*dir;
		eP = start-x*dir;
	}

	float implicit(vec3 r) { return length(f-r) - fabs(dot(eNormal, r-eP)); }

	Hit intersect(Ray const& ray) {
		Hit hit;

		float a = dot(ray.dir,ray.dir);
		float b = 2*dot(ray.dir, ray.start-f);
		float c = dot(ray.start-f,ray.start-f);
		a -= powf(dot(eNormal,ray.dir),2);
		b -= 2*dot(eNormal,ray.dir)*dot(eNormal,ray.start-eP);
		c -= powf(dot(eNormal,ray.start-eP),2);

		std::pair<float,float> sols = quardratic(a,b,c);
		if(isnan(sols.first)) return hit;

		float t = sols.second;
		vec3 position = ray.start+t*ray.dir;
		float distFromStart = dot(position-start, dir);
		if (distFromStart > height) {
			t = sols.first;
			position = ray.start+t*ray.dir;
			distFromStart = dot(position-start, dir);
			if (distFromStart > height) return hit;
		}

		float eps = 0.001;
		float derX = implicit(position + vec3(eps,0,0));
		float derY = implicit(position + vec3(0,eps,0));
		float derZ = implicit(position + vec3(0,0,eps));

		hit.t = t;
		hit.material = material;
		hit.position = position;
		hit.normal = normalize(vec3(derX,derY,derZ));
		if (dot(-ray.dir, hit.normal) < 0) hit.normal = -hit.normal;
 		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
  public:
	void set(vec3 eye, vec3 lookat, vec3 viewUp, float fov) {
		this->eye = eye; this->lookat = lookat; this->fov = fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(viewUp,w))*windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		float ndcx = 2.0f*(X+0.5f)/windowWidth-1;
		float ndcy = 2.0f*(Y+0.5f)/windowHeight-1;
		vec3 dir = (lookat + right*ndcx + up*ndcy) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {
		vec3 d = eye - lookat;
		vec4 tmp = vec4(d.x, d.y, d.z, 1);
		vec4 eye4 = tmp * RotationMatrix(dt, vec3(0,0,1));
		eye = vec3(eye4.x, eye4.y, eye4.z);
		eye = eye + lookat;
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 direction, vec3 Le): direction(normalize(direction)), Le(Le) {}
};

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;

	vec3 viewUp = vec3(0,0,1);
	vec3 lookat = vec3(0,0,0);
	vec3 eye = vec3(7,0,0.8f);
	float fov = 45.0f / 180 * M_PI;
  public:
	void build() {
		camera.set(eye,lookat,viewUp,fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1,1,1), Le(2,2,2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2,2,2);
		Material* materialPlane = new Material(kd1, ks, 50);
		Material* materialLamp = new Material(kd2, ks, 50);

		objects.push_back(new Plane(vec3(0,0,1), vec3(0,0,0), materialPlane));
		objects.push_back(new Sphere(vec3(0,0,0), 0.1f, materialPlane));
		//objects.push_back(new Cylinder(vec3(0,0,1), 2, 1, vec3(0,0,1), materialLamp));
		objects.push_back(new Paraboloid(vec3(0,0,0), vec3(0,0,1), 2, 1, materialLamp));
	}
	
	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);

		// TODO pragma
		for (int Y = 0; Y < windowHeight; Y++) {
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object: objects) {
			Hit hit = object->intersect(ray);
			if(hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object: objects) {
			if(object->intersect(ray).t > 0) return true;
		}
		return false;
	}

	vec3 trace(Ray ray) {
		Hit hit = firstIntersect(ray);
		if(hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light: lights) {
			Ray shadowRay(hit.position + hit.normal * eps, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta <= 0) printf("----%ef\n", cosDelta);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le*hit.material->ks*powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}

	void Animate(float dt) { /*camera.Animate(dt);*/  eye.z += 1; camera.set(eye,lookat,viewUp,fov); }
};

Scene scene;
GPUProgram gpuProgram;

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	layout(location = 0) in vec2 cVP; // c prefix: ndc
	out vec2 uv;

	void main() {
		uv = (cVP + vec2(1,1)) / 2;
		gl_Position = vec4(cVP.xy, 0, 1);
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform sampler2D textureUnit;
	in vec2 uv;
	out vec4 outColor;

	void main() {
		outColor = texture(textureUnit, uv);
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao = 0, textureId = 0;
  public:
	FullScreenTexturedQuad() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vpCoos[] = { -1,-1,  1,-1,  1,1,  -1,1};
		glBufferData(GL_ARRAY_BUFFER, sizeof(vpCoos), vpCoos, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,nullptr);
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //bilineáris szűrés
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad = new FullScreenTexturedQuad();
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth*windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'd': glutPostRedisplay(); break;
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

void onIdle() {
	scene.Animate(0.1f);
	glutPostRedisplay();
}
