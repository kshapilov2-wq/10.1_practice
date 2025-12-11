#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

using namespace std;

using Matrix = vector<vector<double>>;
using Vector = vector<double>;

// ---------------- Вспомогательные функции ----------------

// Создать матрицу заданного размера и заполнить константой
Matrix makeMatrix(int rows, int cols, double value = 0.0) {
	return Matrix(rows, vector<double>(cols, value));
}

// Единичная матрица
Matrix identityMatrix(int n) {
	Matrix I = makeMatrix(n, n, 0.0);
	for (int i = 0; i < n; ++i) I[i][i] = 1.0;
	return I;
}

// Печать матрицы
void printMatrix(const Matrix& A, const string& name) {
	cout << name << " (" << A.size() << "x" << (A.empty() ? 0 : A[0].size()) << "):\n";
	int rows = (int)A.size();
	int cols = rows ? (int)A[0].size() : 0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			cout << setw(12) << A[i][j] << " ";
		}
		cout << "\n";
	}
	cout << "\n";
}

// Печать вектора
void printVector(const Vector& v, const string& name) {
	cout << name << " (" << v.size() << "):\n";
	for (double x : v) cout << setw(12) << x << " ";
	cout << "\n\n";
}

// Умножение матрицы на вектор
Vector matVec(const Matrix& A, const Vector& x) {
	int m = (int)A.size();
	int n = m ? (int)A[0].size() : 0;
	Vector y(m, 0.0);
	for (int i = 0; i < m; ++i) {
		double sum = 0.0;
		for (int j = 0; j < n; ++j) sum += A[i][j] * x[j];
		y[i] = sum;
	}
	return y;
}

// Транспонирование матрицы
Matrix transpose(const Matrix& A) {
	int m = (int)A.size();
	int n = m ? (int)A[0].size() : 0;
	Matrix AT = makeMatrix(n, m, 0.0);
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			AT[j][i] = A[i][j];
	return AT;
}

// Умножение матриц C = A * B
Matrix matMul(const Matrix& A, const Matrix& B) {
	int m = (int)A.size();
	int k = m ? (int)A[0].size() : 0;
	int n = (int)B[0].size();
	Matrix C = makeMatrix(m, n, 0.0);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			double sum = 0.0;
			for (int t = 0; t < k; ++t) sum += A[i][t] * B[t][j];
			C[i][j] = sum;
		}
	}
	return C;
}

// Скалярное произведение
double dot(const Vector& a, const Vector& b) {
	double s = 0.0;
	int n = (int)a.size();
	for (int i = 0; i < n; ++i) s += a[i] * b[i];
	return s;
}

// Евклидова норма вектора
double norm2(const Vector& v) {
	return sqrt(dot(v, v));
}

// Норма Фробениуса матрицы
double frobNorm(const Matrix& A) {
	double s = 0.0;
	for (const auto& row : A)
		for (double x : row) s += x * x;
	return sqrt(s);
}

// Генерация случайной матрицы с равномерным распределением
Matrix randomMatrix(int m, int n, double minVal = -1.0, double maxVal = 1.0) {
	mt19937 gen(0); // фиксированное зерно для воспроизводимости
	uniform_real_distribution<double> dist(minVal, maxVal);
	Matrix A = makeMatrix(m, n, 0.0);
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			A[i][j] = dist(gen);
	return A;
}

// ---------------- Пункт 1: Алгоритм Голуба–Кахана ----------------

struct GKResult {
	Matrix U;      // m x p
	Matrix V;      // n x p
	Matrix B;      // p x p (бидиагональная)
	Vector alpha;  // диагональ B
	Vector beta;   // наддиагональ B (beta[0] — beta1 из алгоритма, в B не входит)
};

GKResult golubKahan(const Matrix& A) {
	int m = (int)A.size();
	int n = m ? (int)A[0].size() : 0;
	int p = min(m, n);

	GKResult res;
	res.U = makeMatrix(m, p, 0.0);
	res.V = makeMatrix(n, p, 0.0);
	res.alpha.assign(p, 0.0);
	res.beta.assign(p, 0.0);

	if (p == 0) {
		res.B = makeMatrix(0, 0, 0.0);
		return res;
	}

	// Стартовый вектор u1 (произвольный ненулевой)
	Vector u1(m, 0.0);
	mt19937 gen(1);
	uniform_real_distribution<double> dist(-1.0, 1.0);
	for (int i = 0; i < m; ++i) u1[i] = dist(gen);

	double beta1 = norm2(u1);
	if (beta1 == 0.0) {
		u1[0] = 1.0;
		beta1 = 1.0;
	}
	for (int i = 0; i < m; ++i) u1[i] /= beta1;

	// u1 -> первый столбец U
	for (int i = 0; i < m; ++i) res.U[i][0] = u1[i];
	res.beta[0] = beta1;

	Matrix AT = transpose(A);

	// Первый правый направляющий вектор
	Vector r = matVec(AT, u1);
	double alpha1 = norm2(r);
	if (alpha1 == 0.0) {
		res.alpha[0] = 0.0;
	}
	else {
		for (int i = 0; i < n; ++i) r[i] /= alpha1;
		for (int i = 0; i < n; ++i) res.V[i][0] = r[i];
		res.alpha[0] = alpha1;
	}

	Vector u_prev = u1;
	Vector v_curr = r;

	// Основной цикл k = 1..p-1
	for (int k = 0; k < p - 1; ++k) {
		// p_vec = A v_k − alpha_k u_k
		Vector Av = matVec(A, v_curr);
		for (int i = 0; i < m; ++i) Av[i] -= res.alpha[k] * u_prev[i];

		double beta_next = norm2(Av);
		res.beta[k + 1] = beta_next;
		if (beta_next == 0.0) break; // ранг меньше p

		Vector u_next(m, 0.0);
		for (int i = 0; i < m; ++i) u_next[i] = Av[i] / beta_next;
		for (int i = 0; i < m; ++i) res.U[i][k + 1] = u_next[i];

		// r = A^T u_{k+1} − beta_{k+1} v_k
		r = matVec(AT, u_next);
		for (int i = 0; i < n; ++i) r[i] -= beta_next * v_curr[i];

		double alpha_next = norm2(r);
		res.alpha[k + 1] = alpha_next;
		if (alpha_next == 0.0) break;

		Vector v_next(n, 0.0);
		for (int i = 0; i < n; ++i) v_next[i] = r[i] / alpha_next;
		for (int i = 0; i < n; ++i) res.V[i][k + 1] = v_next[i];

		u_prev = u_next;
		v_curr = v_next;
	}

	// Возможное уменьшение p, если ранг меньше
	int p_eff = p;
	for (int i = p - 1; i >= 0; --i) {
		if (res.alpha[i] != 0.0 || (i + 1 < p && res.beta[i + 1] != 0.0)) {
			p_eff = i + 1;
			break;
		}
	}
	if (p_eff < p) {
		res.U.resize(m);
		for (int i = 0; i < m; ++i) res.U[i].resize(p_eff);
		res.V.resize(n);
		for (int i = 0; i < n; ++i) res.V[i].resize(p_eff);
		res.alpha.resize(p_eff);
		res.beta.resize(p_eff);
		p = p_eff;
	}

	// Явная бидиагональная матрица B
	res.B = makeMatrix(p, p, 0.0);
	for (int i = 0; i < p; ++i) {
		res.B[i][i] = res.alpha[i];
		if (i + 1 < p) res.B[i][i + 1] = res.beta[i + 1];
	}

	return res;
}

// ---------------- Пункт 2: Диагонализация симметричной матрицы (Якоби) ----------------

void jacobiEigenSymmetric(Matrix A, Matrix& Q, Vector& eigs) {
	int n = (int)A.size();
	Q = identityMatrix(n);
	const int maxIter = 100 * n * n;
	const double tol = 1e-12;

	for (int iter = 0; iter < maxIter; ++iter) {
		// Максимальный по модулю внедиагональный элемент
		int p = 0, q = 1;
		double maxOff = 0.0;
		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				double val = fabs(A[i][j]);
				if (val > maxOff) {
					maxOff = val;
					p = i;
					q = j;
				}
			}
		}
		if (maxOff < tol) break;

		double app = A[p][p];
		double aqq = A[q][q];
		double apq = A[p][q];

		double phi = 0.5 * atan2(2.0 * apq, aqq - app);
		double c = cos(phi);
		double s = sin(phi);

		// Обновление A = J^T*A*J, J — вращение в плоскости (p,q)
		for (int k = 0; k < n; ++k) {
			double aik = A[p][k];
			double aqk = A[q][k];
			A[p][k] = c * aik - s * aqk;
			A[q][k] = s * aik + c * aqk;
		}
		for (int k = 0; k < n; ++k) {
			double kip = A[k][p];
			double kiq = A[k][q];
			A[k][p] = c * kip - s * kiq;
			A[k][q] = s * kip + c * kiq;
		}

		// Обновляем матрицу собственных векторов Q = Q * J
		for (int k = 0; k < n; ++k) {
			double qkp = Q[k][p];
			double qkq = Q[k][q];
			Q[k][p] = c * qkp - s * qkq;
			Q[k][q] = s * qkp + c * qkq;
		}
	}

	eigs.assign(n, 0.0);
	for (int i = 0; i < n; ++i) eigs[i] = A[i][i];

	// Сортировка собственных значений и соответствующих векторов по убыванию
	vector<int> idx(n);
	for (int i = 0; i < n; ++i) idx[i] = i;
	sort(idx.begin(), idx.end(), [&](int i, int j) { return eigs[i] > eigs[j]; });

	Vector eigs_sorted(n);
	Matrix Q_sorted = makeMatrix(n, n, 0.0);
	for (int k = 0; k < n; ++k) {
		int j = idx[k];
		eigs_sorted[k] = eigs[j];
		for (int i = 0; i < n; ++i) Q_sorted[i][k] = Q[i][j];
	}

	eigs = eigs_sorted;
	Q = Q_sorted;
}

// -------- SVD через A^T*A (точное разложение) --------

void svdViaATA(const Matrix& A, Matrix& U, Vector& sigma, Matrix& V) {
	int m = (int)A.size();
	int n = m ? (int)A[0].size() : 0;
	Matrix AT = transpose(A);
	Matrix ATA = matMul(AT, A);       // n x n

	Matrix Qe;                        // собственные векторы ATA
	Vector lambda;                    // собственные значения ATA
	jacobiEigenSymmetric(ATA, Qe, lambda);

	int p = min(m, n);                // число сингулярных чисел
	sigma.clear();
	U = makeMatrix(m, 0, 0.0);
	V = makeMatrix(n, 0, 0.0);

	// Берем первые p положительных собственных значений
	for (int k = 0; k < (int)lambda.size() && (int)sigma.size() < p; ++k) {
		double lam = lambda[k];
		if (lam < 0.0) lam = 0.0;
		double s = sqrt(lam);
		if (s < 1e-12) continue; // нулевые / слишком маленькие пропускаем

		sigma.push_back(s);

		// Правый сингулярный вектор v_k — столбец Qe
		int newIndex = (int)sigma.size() - 1;
		if ((int)V[0].size() < (int)sigma.size()) {
			// расширяем V по столбцам
			for (int i = 0; i < n; ++i) V[i].push_back(0.0);
		}
		for (int i = 0; i < n; ++i) V[i][newIndex] = Qe[i][k];

		// Левый сингулярный вектор u_k = (1 / s) * A * v_k
		Vector vk(n);
		for (int i = 0; i < n; ++i) vk[i] = V[i][newIndex];
		Vector uk = matVec(A, vk);

		if ((int)U[0].size() < (int)sigma.size()) {
			// расширяем U по столбцам
			for (int i = 0; i < m; ++i) U[i].push_back(0.0);
		}
		for (int i = 0; i < m; ++i) U[i][newIndex] = uk[i] / s;
	}
}

// ---------------- Пункты 3–4: Демонстрация, SVD и оценка погрешности ----------------

int main() {
	setlocale(LC_ALL, "Russian");
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	cout << fixed << setprecision(6);

	int m, n;
	cout << "Введите размеры матрицы A (m n): ";
	if (!(cin >> m >> n)) {
		cout << "Некорректный ввод.\n";
		return 0;
	}
	if (m <= 0 || n <= 0) {
		cout << "Размеры должны быть положительными.\n";
		return 0;
	}

	cout << "\n============================================================\n";
	cout << "ПУНКТ 0. Исходная матрица A\n";
	cout << "============================================================\n";

	Matrix A = randomMatrix(m, n, -1.0, 1.0);
	printMatrix(A, "Исходная матрица A");

	cout << "\n============================================================\n";
	cout << "ПУНКТ 1. Бидиагонализация (алгоритм Голуба–Кахана)\n";
	cout << "============================================================\n";

	GKResult gk = golubKahan(A);
	printMatrix(gk.U, "Матрица U (Голуб–Кахан)");
	printMatrix(gk.B, "Бидиагональная матрица B");
	printMatrix(gk.V, "Матрица V (Голуб–Кахан)");

	int p = (int)gk.B.size();

	cout << "\n============================================================\n";
	cout << "ПУНКТ 2. Построение T = B^T * B и диагонализация методом Якоби\n";
	cout << "============================================================\n";

	Matrix BT = transpose(gk.B);
	Matrix T = matMul(BT, gk.B);
	printMatrix(T, "Матрица T = B^T * B");

	Matrix Qe_T;
	Vector lambda_T;
	jacobiEigenSymmetric(T, Qe_T, lambda_T);
	printMatrix(Qe_T, "Матрица собственных векторов Q_e (для T)");
	printVector(lambda_T, "Собственные значения T (приближенно sigma_i^2)");

	cout << "\n============================================================\n";
	cout << "ПУНКТ 3. Сингулярные числа и сингулярные векторы (точное SVD через A^T*A)\n";
	cout << "============================================================\n";

	Matrix U_svd, V_svd;
	Vector sigma;
	svdViaATA(A, U_svd, sigma, V_svd);

	printVector(sigma, "Сингулярные числа sigma_i (из A^T*A)");
	printMatrix(U_svd, "U (левые сингулярные векторы A)");
	printMatrix(V_svd, "V (правые сингулярные векторы A)");

	cout << "\n============================================================\n";
	cout << "ПУНКТ 4. Восстановление A и оценка точности SVD\n";
	cout << "============================================================\n";

	int r = (int)sigma.size();
	Matrix SigmaMat = makeMatrix(r, r, 0.0);
	for (int i = 0; i < r; ++i) SigmaMat[i][i] = sigma[i];

	Matrix US = matMul(U_svd, SigmaMat);   // m x r
	Matrix Vt = transpose(V_svd);          // r x n
	Matrix A_approx = matMul(US, Vt);      // m x n

	printMatrix(A_approx, "Восстановленная матрица A_approx по SVD");

	Matrix diff = A;
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			diff[i][j] -= A_approx[i][j];

	double normA = frobNorm(A);
	double normDiff = frobNorm(diff);

	cout << "||A||_F = " << normA << "\n";
	cout << "||A - A_approx||_F = " << normDiff << "\n";
	cout << "Относительная погрешность по норме Фробениуса: "
		<< (normA > 0 ? normDiff / normA : 0.0) << "\n\n";

	cout << "Проверка сингулярных пар (невязка (A^T*A*v - (sigma^2)*v)):\n";
	Matrix AT = transpose(A);
	Matrix ATA = matMul(AT, A);
	for (int k = 0; k < r; ++k) {
		Vector vk(n, 0.0);
		for (int i = 0; i < n; ++i) vk[i] = V_svd[i][k];

		Vector ATA_vk = matVec(ATA, vk);
		Vector rhs = vk;
		double sigma2 = sigma[k] * sigma[k];
		for (int i = 0; i < n; ++i) rhs[i] *= sigma2;

		for (int i = 0; i < n; ++i) ATA_vk[i] -= rhs[i];
		double resNorm = norm2(ATA_vk);
		cout << "Невязка сингулярной пары " << k + 1 << ": " << resNorm << "\n";
	}

	return 0;
}
