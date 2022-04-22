#include "framework.h"
#include <bits/stdc++.h>

struct Material {
	// visszaverző képesség (3 hullámhosszon): ambiens, diffúz, spekuláris
	vec3 ka, kd, ks;
	// shine érték
	float shininess;
	// TODO miért *M_PI???XDD
	Material(vec3 kd, vec3 ks, float shininess): ka(kd*M_PI), kd(kd), ks(ks), shininess(shininess) {}
};

// sugár és felület leírása: ezek kellenek róla
struct Hit {
	float t;
	// világ koo-k
	vec3 position, normal;
	// Felületnél nem feltétlen homogén, lásd textúrázás
	Material* material;
	Hit() { t = -1; }
};

// sugár: félegyenes
struct Ray {
	vec3 start, dir;
	// dir normalizálás
	Ray(vec3 start, vec3 dir): start(start), dir(normalize(dir)) {}
};

//base class
class Intersectable {
  protected:
	Material* material;

	Intersectable(Material* material): material(material) {}
  public:
	// t==-1 => nincs találat
	virtual Hit intersect(Ray const& ray) = 0;
};

class Sphere: public Intersectable {
	vec3 center;
	float radius;
  public:
	// TODO center why ref
    Sphere(vec3& center, float radius, Material* material): Intersectable(material), center(center), radius(radius) {}

	Hit intersect(Ray const& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(ray.dir, dist) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b*b-4.0f*a*c;
		// no intersection
		if (discr < 0) return hit; // default value is -1
		float sqrtDiscr = sqrtf(discr);
		// t1 >= t2
		float t1 = (-b+sqrtDiscr)/2.0f/a;
		float t2 = (-b-sqrtDiscr)/2.0f/a;
		// t1 behind camera => t2 also
		if (t1 <= 0) return hit;
		// by default t2 is closer to camera;
		hit.t = t2>0 ? t2 : t1;

		// t alapján visszahelyettesítés
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;
		hit.material = material;
		return hit;
	}
};

class Camera {
	// lookat: animáció mit nézzen
	vec3 eye, lookat, right, up;
	// "Computed value", kamera animációhoz kell
	float fov;
  public:
    // kamera beállítása: animáció miatt kell frissíteni a right és upot !!!!!!
	//	viewUp: preferált függőleges irány
	//	fov: kamera függőleges látószöge (négyzet => vízszintes is)
	void set(vec3 eye, vec3 lookat, vec3 viewUp, float fov) {
		this->eye = eye; this->lookat = lookat; this->fov = fov;
		
		// ortogonalizációs lépés: viewUp megkötés miatt
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(viewUp,w)*windowSize);
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		// 0.5f: center of a pixel
		float ndcx = 2*(X+0.5f)/windowWidth-1;
		float ndcy = 2*(Y+0.5f)/windowHeight-1;
		// eye to view panel ndc coos
		vec3 dir = (lookat + right*ndcx + up*ndcy) - eye;
		return Ray(eye, dir);
	}

	// lookat ponton átmenő függőleges egyenes körül forgatás
	// tekintet folyamastosan lookat ponton
	void Animate(float dt) {
		// lookat a pivot point !!!!!!!
		vec3 d = eye - lookat;
		// TODO Rotation matrix
		eye = vec3(d.x * cosf(dt) + d.z * sinf(dt), d.y, -d.x * sin(dt) + d.z * cos(dt));
		// pivot point visszatolás !!!!
		eye = eye + lookat;
		set(eye, lookat, up, fov);
	}
};

// irányfényforrás
struct Light {
	vec3 direction;
	vec3 Le; // intezitás: spektruma (3 hullámhossz)
	Light(vec3 direction, vec3 Le): Le(Le) { this->Le = normalize(Le); } //!!!!
};

float rnd() { return (float)rand() / RAND_MAX; }
float const eps = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	// ambiens fény
	vec3 La;
  public:
	// virtuális világ felépítése: elemek elhelyezése
	// TODO kísérletezés
	void build() {
		vec3 eye = vec3(0,0,2), viewUp = vec3(0,1,0), lookat = vec3(0,0,0);
		float fov = 45 * M_PI / 180;
		camera.set(eye,lookat,viewUp,fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1,1,1), Le(2,2,2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2,2,2);
		Material* material1 = new Material(kd1, ks, 50);
		Material* material2 = new Material(kd2, ks, 50);

		for (int i = 0; i < 150; i++) {
			// TODO más eltolás is jó?
			vec3 pos1 = vec3(rnd()-0.5f, rnd()-0.5f, rnd()-0.5f);
			vec3 pos2 = vec3(rnd()-0.5f, rnd()-0.5f, rnd()-0.5f);
			objects.push_back(new Sphere(pos1, rnd()*0.1f, material1));
			objects.push_back(new Sphere(pos2, rnd()*0.1f, material2));
		}
	}

	// lefényképezés
	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
		
		printf("Rtime %d ms\n",glutGet(GLUT_ELAPSED_TIME)-timeStart);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object: objects) {
			Hit hit = object->intersect(ray);
			// eddig még nem volt valid min eset kezelése
			if(hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		// normál vektor felénk mutasson !!!!!!!!!!!!!
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
			// fényforrás látható-e a szemből látható pontban!!
			// TODO kísérlet: törlés vagy eps nélkül
			// normál vektor kifele mutat
			// light->direction: fény fele mutat
			Ray shadowRay(hit.position + hit.normal * eps, light->direction);
			float cosTheta = dot(hit.normal, light->direction); // egységvektorok !!!!
			// cos negatív: ez az obj árnyékol maga
			// más obj takar-e
			// shadowIntersect: csak a visszatérés miatt új fgv
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				// V = -ray.dir !!!!
				// egységvektorok !!!!
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway); //egységvektorok !!!!
				// TODO ??
				//std::assert(cosDelta > 0);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le*hit.material->ks*powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
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
		uv = (cVP + vec(1,1)) / 2;
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

// téglalap: ndc-beli teljesen kitöltve, textúrázás is
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

		// EGÉSZ!!!!
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		// TODO miért kell 2? honnan tudja mikor van min és mag
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
			// uniform beáll
			glUniform1i(location, textureUnit);
			// textúra mintavevő egység akitválása
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

	//glClearColor(0, 0, 0, 0);
	//glClear(GL_COLOR_BUFFER_BIT);
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {

	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

void onMouse(int button, int state, int pX, int pY) {

	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

void onIdle() {
	scene.Animate(0.1f);
	glutPostRedisplay(); //!!!!
	//long time = glutGet(GLUT_ELAPSED_TIME);
}
