#include "framework.h"

float floatEqual(float subject, float number) {
	float eps = 0.001;
	return subject > number - eps && subject < number + eps;
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

float quardraticSmaller(float a, float b, float c) {
	float discr = b*b-4.0f*a*c;
	if (discr < 0) return nanf("");
	float sqrtDiscr = sqrtf(discr);
	float t1 = (-b+sqrtDiscr)/2.0f/a;
	float t2 = (-b-sqrtDiscr)/2.0f/a;
	if (t1 <= 0) return nanf("");
	return (t2>0) ? t2 : t1;
}

class Sphere: public Intersectable {
	vec3 center;
	float radius;
  public:
    Sphere(vec3 center, float radius, Material* material): Intersectable(material), center(center), radius(radius) {}

	Hit intersect(Ray const& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(ray.dir, dist) * 2.0f;
		float c = dot(dist, dist) - radius * radius;

		float sol = quardraticSmaller(a,b,c);
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
	
	Plane(vec3 normal, vec3 point, Material* material): Intersectable(material), normal(normalize(normal)), point(point) {}

	Hit intersect(Ray const& ray) {
		Hit hit;
		float divisor = dot(ray.dir, normal);
		if (floatEqual(divisor, 0)) return hit;
		float t = - dot(ray.start, normal) / divisor;
		if (t < 0) return hit;
		hit.position = ray.start + t*ray.dir;
		hit.normal = normal;
		hit.material = material;
		hit.t = t;
		return hit;
	}
};

class Cylinder: public Intersectable {
	vec3 start;
	float height;
	vec3 dir;
	vec3 normal;
	float r;

	std::vector<Plane> planes;

  public:
	Cylinder(vec3 start, float height, float r, vec3 dir, Material* material): Intersectable(material), start(start), height(height), r(r), dir(normalize(dir)) {
		normal.x = 1;
		normal.y = 0;
		normal.z = -dir.x / dir.z;
		normal = normalize(normal);

		planes.push_back(Plane(dir, start, material));
		planes.push_back(Plane(dir, start + dir * height, material));
	}

	Hit intersect(Ray const& ray) {
		Hit hit;

		for (Plane plane: planes) {
			Hit planeHit = plane.intersect(ray);
			if (length(planeHit.position - plane.point) <= r) {
				return planeHit;
			}
		}

		// TODO use multiplication for simpler form
		float a = powf(normal.x * ray.dir.x, 2) + powf(normal.y * ray.dir.y, 2);
		float b = 2*normal.x*ray.dir.x*(ray.start.x - start.x) + 2*normal.y*ray.dir.y*(ray.start.y - start.y);
		float c = powf(normal.x*(ray.start.x-start.x), 2) + powf(normal.y*(ray.start.y-start.y), 2) - powf(r,2);

		float sol = quardraticSmaller(a,b,c);
		if (isnan(sol)) return hit;

		vec3 position = ray.start + ray.dir * hit.t;
		float distanceFromStart = dot(dir, position - start);
		if (distanceFromStart > height || distanceFromStart < 0) return hit;

		vec3 center = start + distanceFromStart*dir;

		hit.t = sol;
		hit.material = material;
		hit.normal = (position - center) / r;
		hit.position = position;
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

float rnd() { return (float)rand() / RAND_MAX; }
float const eps = 0.0001f;

void print(std::string text, vec3 v) {
	printf("%s: %ef %ef %ef\n", text.c_str(), v.x, v.y, v.z);
}

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;

	vec3 viewUp = vec3(0,0,1);
	vec3 lookat = vec3(0,0,0);
	vec3 eye = vec3(5,0,0.8f);
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
		objects.push_back(new Sphere(vec3(0,0,0), 0.1f, materialLamp));
		objects.push_back(new Cylinder(vec3(0,0,0), 10, 1, vec3(0,0,1), materialLamp));
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
