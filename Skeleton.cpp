#include "framework.h"

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
    Sphere(vec3 center, float radius, Material* material): Intersectable(material), center(center), radius(radius) {}

	Hit intersect(Ray const& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(ray.dir, dist) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b*b-4.0f*a*c;
		if (discr < 0) return hit;
		float sqrtDiscr = sqrtf(discr);
		float t1 = (-b+sqrtDiscr)/2.0f/a;
		float t2 = (-b-sqrtDiscr)/2.0f/a;
		if (t1 <= 0) return hit;
		hit.t = (t2>0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
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
		eye = vec3(eye4.x, eye4.y, eye4.z) / eye4.w;
		eye = eye + lookat;
		set(eye, lookat, up, fov);
	}
};
struct Light {
	vec3 direction;
	vec3 Le; // intezitás: spektruma (3 hullámhossz)
	Light(vec3 direction, vec3 Le): direction(normalize(direction)), Le(Le) {} //!!!!
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

	vec3 viewUp = vec3(-1,0,0);
	vec3 lookat = vec3(0,0,-1);
	vec3 eye = vec3(2,0,0);
	float fov = 45 * M_PI / 180;
  public:
	void build() {
		camera.set(eye,lookat,viewUp,fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1,1,1), Le(2,2,2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2,2,2);
		Material* material1 = new Material(kd1, ks, 50);
		Material* material2 = new Material(kd2, ks, 50);

		for (int i = 0; i < 10; i++) {
			vec3 pos1 = vec3(rnd()-0.5f, rnd()-0.5f, rnd()-0.5f);
			vec3 pos2 = vec3(rnd()-0.5f, rnd()-0.5f, rnd()-0.5f);
			objects.push_back(new Sphere(pos1, rnd()*0.1f, material1));
			objects.push_back(new Sphere(pos2, rnd()*0.1f, material2));
		}
	}
	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
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

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if(hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light: lights) {
			Ray shadowRay(hit.position + hit.normal * eps, light->direction); // eps!!!
			float cosTheta = dot(hit.normal, light->direction); // egységvektorok !!!!
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway); //egységvektorok !!!!
				if (cosDelta <= 0) printf("----%ef\n", cosDelta);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le*hit.material->ks*powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}

	void translate(float dt) {
		print("viewUp",viewUp);
		vec4 viewUp4 = vec4(viewUp.x,viewUp.y,viewUp.z);
		viewUp4 = viewUp4 * RotationMatrix(dt, vec3(0,1,0));
		viewUp = vec3(viewUp4.x, viewUp4.y, viewUp4.z);
		print("viewUp",viewUp);

		print("lookat",lookat);
		vec4 lookat4 = vec4(lookat.x,lookat.y,lookat.z);
		lookat4 = lookat4 * RotationMatrix(dt, vec3(0,1,0));
		lookat = vec3(lookat4.x, lookat4.y, lookat4.z);
		print("lookat",lookat);
		
		camera.set(eye,lookat,viewUp,fov);
	}

	void Animate(float dt) { camera.Animate(dt); }
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
	return;
	switch (key) {
	case 't': scene.translate(0.1f); break;
	case 'g': scene.translate(-0.1f); break;
	}
	
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

void onIdle() {
	return;
	scene.Animate(0.1f);
	glutPostRedisplay(); //!!!!
}
