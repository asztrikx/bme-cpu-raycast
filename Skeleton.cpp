//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Vörös Asztrik
// Neptun : WYZJ90
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

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

std::pair<float,float> quardratic(float a, float b, float c) {
	float discr = b*b-4.0f*a*c;
	if (discr < 0) return std::make_pair(nanf(""),nanf(""));
	float sqrtDiscr = sqrtf(discr);
	float t1 = (-b+sqrtDiscr)/2.0f/a;
	float t2 = (-b-sqrtDiscr)/2.0f/a;
	return std::make_pair(t1,t2);
}

float rnd() { return (float)rand() / RAND_MAX; }

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
  protected:
	Material * material;
  public:
	virtual Hit intersect(const Ray& ray) = 0;
	virtual ~Intersectable() {};
};

class Sphere: public Intersectable {
	vec3 center;
	float radius;
  public:
    Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(Ray const& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(ray.dir, dist) * 2.0f;
		float c = dot(dist, dist) - radius * radius;

		auto sols = quardratic(a,b,c);
		if (isnan(sols.first)) return hit;
		float t1 = sols.first;
		float t2 = sols.second;

		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;
		hit.material = material;
		return hit;
	}
};

struct Plane: public Intersectable {
	vec3 normal;
	vec3 point;
	
	Plane(vec3 _normal, vec3 _point, Material* _material) {
		normal = normalize(_normal);
		point = _point;
		material = _material;
	}

	Hit intersect(Ray const& ray) {
		Hit hit;
		float divisor = dot(ray.dir, normal);
		if (floatEqual(divisor, 0.0f)) return hit;
		float t = dot(point - ray.start, normal) / divisor;
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
	float r;
	vec3 dir90;

	std::vector<Plane> planes;

  public:
	Cylinder(vec3 _start, float _height, float _r, vec3 _dir, Material* _material) {
		start = _start;
		height = _height;
		r = _r;
		dir = normalize(_dir);
		material = _material;

		dir90 = perpendicular(dir);
		planes.push_back(Plane(dir, start + dir * height, material));
		planes.push_back(Plane(dir, start, material));
	}

	Hit intersect(Ray const& ray) {
		Hit hit;
		Hit hitPlane;

		for (Plane plane: planes) {
			Hit hitPlaneCandidate = plane.intersect(ray);
			if (hitPlaneCandidate.t > 0 && length(hitPlaneCandidate.position - plane.point) <= r) {
				if (hitPlane.t < 0 || hitPlaneCandidate.t < hitPlane.t) {
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

		float t = a > 0 ? sols.second : sols.first;
		vec3 position = ray.start + ray.dir * t;
		float distFromStart = dot(dir, position - start);
		if (t < 0 || distFromStart > height || distFromStart < 0) {
			t = a > 0 ? sols.first : sols.second;
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
		return hit;
	}
};

class Paraboloid: public Intersectable {
	vec3 start, dir;
	vec3 eNormal, eP;
	float height;

 public:
	vec3 f;
	Paraboloid(vec3 _start, vec3 _dir, float _height, float fDist, Material* _material) {
		start = _start;
		dir = normalize(_dir);
		height = _height;
		material = _material;

		eNormal = dir;
		eP = start-fDist*dir;
		f = start+fDist*dir;
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
		if (t < 0 || distFromStart > height ) {
			t = sols.first;
			position = ray.start+t*ray.dir;
			distFromStart = dot(position-start, dir);
			if (t < 0 || distFromStart > height ) return hit;
		}

		float eps = 0.001;
		float derX = implicit(position + vec3(eps,0,0));
		float derY = implicit(position + vec3(0,eps,0));
		float derZ = implicit(position + vec3(0,0,eps));

		hit.t = t;
		hit.material = material;
		hit.position = position;
		hit.normal = normalize(vec3(derX,derY,derZ));
 		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
  public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, normalize(dir));
	}
};

class Light {
  protected:
	vec3 Le;
  public:
	Light(vec3 _Le) {
		Le = _Le;
	}
	virtual vec3 getLe(vec3 hitPos) = 0;
	virtual vec3 getDirection(vec3 hitPos) = 0;
	virtual vec3 getPosition() = 0;
	virtual ~Light() {};
};

struct PositionalLight: public Light {
	vec3 point;

	PositionalLight(vec3 _point, vec3 _Le): Light(_Le) {
		point = _point;
	}

	vec3 getDirection(vec3 hitPos) {
		return normalize(point - hitPos);
	}
	vec3 getLe(vec3 hitPos) {
		float r = length(point - hitPos);
		return Le/r/r;
	}
	vec3 getPosition() {
		return point;
	}
};

struct DirectLight: public Light {
	vec3 direction;

	DirectLight(vec3 _direction, vec3 _Le): Light(_Le) {
		direction = normalize(_direction);
	}

	vec3 getDirection(vec3 hitPos) {
		return direction;
	}
	vec3 getLe(vec3 hitPos) {
		return Le;
	}
	vec3 getPosition() {
		return vec3(nanf(""),nanf(""),nanf(""));
	}
};

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La = vec3(0.1f, 0.1f, 0.1f);

	vec3 viewUp = vec3(0,0,1);
	vec3 lookat = vec3(1,0,0);
	vec3 eye = vec3(7,0,5);
	float fov = 45.0f / 180 * M_PI;
	float epsilon = 0.0001;

	float bigCylinderH = 0.1;
	float bigCylinderR = 0.5;
	float sphereR = 1.0f/8;
	float cylinderR = sphereR / 3;
	float paraH = 0.5, paraF = cylinderR + 0.1;
	float cylinderH0 = 2;
	float cylinderH1 = 1;
	vec3 dir0 = normalize(vec3(1,1,2));
	vec3 dir1 = normalize(vec3(-0.5,-1,2.8));
	vec3 paraDir = vec3(-2,-2,1);
	vec3 rot0 = normalize(vec3(1,1,1.5));
	vec3 rot1 = normalize(vec3(2,1,2));
	vec3 rot2 = normalize(vec3(2,2,1));
	vec3 joint1;
	vec3 joint2;
	vec3 joint0 = vec3(0,0,bigCylinderH);
	vec3 kd1 = vec3(55, 60, 63)/255.0f, kd2 = vec3(110, 76, 67)/255.0f;
	Material* materialLamp = new Material(kd1, vec3(2,2,2), 50);
	Material* materialPlane = new Material(kd2, vec3(0.1f,0.1f,0.1f), 50);
  public:
	void recalc() {
		for (int i = 0; i < objects.size(); i++) {
			delete objects[i];
		}
		objects.clear();
		for (int i = 0; i < lights.size(); i++) {
			delete lights[i];
		}
		lights.clear();

		vec3 joint1 = joint0+cylinderH0*dir0;
		vec3 joint2 = joint1+cylinderH1*dir1;

		objects.push_back(new Plane(vec3(0,0,1), vec3(0,0,0), materialPlane));
		objects.push_back(new Cylinder(vec3(0,0,0), bigCylinderH, bigCylinderR, vec3(0,0,1), materialLamp));
		objects.push_back(new Sphere(joint0, sphereR, materialLamp));
		objects.push_back(new Sphere(joint1, sphereR, materialLamp));
		objects.push_back(new Sphere(joint2, sphereR, materialLamp));
		objects.push_back(new Cylinder(joint0, cylinderH0, cylinderR, dir0, materialLamp));
		objects.push_back(new Cylinder(joint1, cylinderH1, cylinderR, dir1, materialLamp));
		objects.push_back(new Paraboloid(joint2 , paraDir, paraH, paraF, materialLamp));

		vec3 lightDirection(1,1,1);
		lights.push_back(new DirectLight(lightDirection, vec3(0.2f,0.2f,0.2f)));
		lights.push_back(new PositionalLight(joint2 + paraF*paraDir, vec3(20,20,20)));

		camera.set(eye,lookat,viewUp,fov);
	}

	void Animate(float dt) {
		vec4 t;

		t = vec4(dir0.x,dir0.y,dir0.z,1) * RotationMatrix(3*dt, rot0);
		dir0 = vec3(t.x, t.y, t.z);

		t = vec4(dir1.x,dir1.y,dir1.z,1) * RotationMatrix(-2*dt, rot1);
		dir1 = vec3(t.x, t.y, t.z);
		
		t = vec4(paraDir.x,paraDir.y,paraDir.z,1) * RotationMatrix(-3*dt, dir1);
		paraDir = vec3(t.x, t.y, t.z);

		eye = eye - lookat;
		t = vec4(eye.x,eye.y,eye.z,1) * RotationMatrix(3*dt, vec3(0,0,1));
		eye = vec3(t.x, t.y, t.z);
		eye = eye+lookat;

		recalc();
	}

	void build() {
		recalc();
	}
	
	void render(std::vector<vec4>& image) {
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
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray, vec3 lightPosition) {
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (isnan(lightPosition.x) || length(ray.start-lightPosition) > length(ray.start-hit.position))) return true;
		}
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			vec3 direction = light->getDirection(hit.position);
			vec3 Le = light->getLe(hit.position);
			Ray shadowRay(hit.position + hit.normal * epsilon, direction);
			float cosTheta = dot(hit.normal, direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay, light->getPosition())) {
				outRadiance = outRadiance + Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}

};

GPUProgram gpuProgram;
Scene scene;

const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;
	out vec4 fragmentColor;

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao, textureId = 0;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight)	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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
		if(location>=0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0+textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth*windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	scene.Animate(0.1*2);
	glutPostRedisplay();
}
