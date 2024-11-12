using System;

class ModifiedNewtonSolver
{
    static double Function1(double x, double y) => Math.Sin(2 * x - y) - 1.2 * x - 0.4;
    static double Function2(double x, double y) => 0.8 * x * x + 1.5 * y * y - 1;

    static double DerivativeF1X(double x, double y) => 2 * Math.Cos(2 * x - y) - 1.2;
    static double DerivativeF1Y(double x, double y) => -Math.Cos(2 * x - y);
    static double DerivativeF2X(double x, double y) => 1.6 * x;
    static double DerivativeF2Y(double x, double y) => 3 * y;

    static void ComputeJacobian(double x, double y, double[,] jacobian)
    {
        jacobian[0, 0] = DerivativeF1X(x, y);
        jacobian[0, 1] = DerivativeF1Y(x, y);
        jacobian[1, 0] = DerivativeF2X(x, y);
        jacobian[1, 1] = DerivativeF2Y(x, y);
    }

    static void SolveLinearSystem(double[,] matrix, double[] results)
    {
        double determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
        double inverseDeterminant = 1 / determinant;

        double tempX = (results[0] * matrix[1, 1] - results[1] * matrix[0, 1]) * inverseDeterminant;
        double tempY = (matrix[0, 0] * results[1] - matrix[1, 0] * results[0]) * inverseDeterminant;

        results[0] = tempX;
        results[1] = tempY;
    }

    static (double, double) ModifiedNewtonIteration(double x, double y)
    {
        double[,] jacobian = new double[2, 2];
        ComputeJacobian(x, y, jacobian);

        double[] functions = { -Function1(x, y), -Function2(x, y) };
        SolveLinearSystem(jacobian, functions);

        double stepSize = 1.0;
        double newX = x + stepSize * functions[0];
        double newY = y + stepSize * functions[1];

        while (Math.Abs(Function1(newX, newY)) > Math.Abs(Function1(x, y)) ||
               Math.Abs(Function2(newX, newY)) > Math.Abs(Function2(x, y)))
        {
            stepSize *= 0.5;
            newX = x + stepSize * functions[0];
            newY = y + stepSize * functions[1];
            if (stepSize < 1e-8) break; 
        }

        return (newX, newY);
    }

    static void RunModifiedNewtonMethod(double x, double y, int maxIterations)
    {
        Console.WriteLine("This program is solving the system of non-linear equations:");
        Console.WriteLine("sin(2x - y) - 1.2x = 0.4 and 0.8x^2 + 1.5y^2 = 1 using the Modified Newton method.");

        for (int i = 1; i <= maxIterations; i++)
        {
            (x, y) = ModifiedNewtonIteration(x, y);
            Console.WriteLine($"Iteration {i}: x = {x}, y = {y}");
        }

        Console.WriteLine($"Final result: x = {x}, y = {y}");
    }
    ///////////////////////
    static void ReadMatrix(double[,] m, int N)
    {
        for (int i = 1; i <= N; i++)
        {
            for (int j = 1; j <= N; j++)
            {
                m[i, j] = double.Parse(Console.ReadLine());
            }
        }
    }

    static void WriteMatrix(double[,] m, int N)
    {
        for (int i = 1; i <= N; i++)
        {
            for (int j = 1; j <= N; j++)
            {
                Console.Write((m[i, j] == 0 ? 0 : m[i, j]) + " ");
            }
            Console.WriteLine();
        }
    }

    static bool IsSymmetric(double[,] m, int N)
    {
        for (int i = 1; i < N; i++)
        {
            for (int j = i + 1; j <= N; j++)
            {
                if (m[i, j] != m[j, i])
                    return false;
            }
        }
        return true;
    }

    static void RewriteSquare(double[,] m, double[,] m2, int N)
    {
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                m2[i, j] = m[i, j];
    }

    static void RewriteString(double[] m, double[] m2, int N)
    {
        for (int i = 1; i <= N; i++)
            m2[i] = m[i];
    }

    static void Reset(double[,] u, double[,] u_t, int N)
    {
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
            {
                u[i, j] = 0;
                u_t[i, j] = 0;
            }
    }

    static void MaxSearch(double[,] m, int N, out int imax, out int jmax)
    {
        imax = 1;
        jmax = 2;
        for (int i = 1; i <= N; i++)
        {
            for (int j = 1; j <= N; j++)
            {
                if (i != j && m[i, j] > m[imax, jmax])
                {
                    imax = i;
                    jmax = j;
                }
            }
        }
    }

    static void MultiplySquare(double[,] m1, double[,] m2, double[,] ans, int N)
    {
        for (int i = 1; i <= N; i++)
        {
            for (int j = 1; j <= N; j++)
            {
                double sum = 0;
                for (int k = 1; k <= N; k++)
                    sum += m1[i, k] * m2[k, j];
                ans[i, j] = sum;
            }
        }
    }

    static void MultiplyString(double[,] m1, double[] m2, double[] ans, int N)
    {
        for (int i = 1; i <= N; i++)
        {
            ans[i] = 0;
            for (int j = 1; j <= N; j++)
                ans[i] += m1[i, j] * m2[j];
        }
    }

    static double Norm(double[,] m, int N)
    {
        double etalon = 0;
        for (int i = 1; i <= N; i++)
        {
            double sum = 0;
            for (int j = 1; j <= N; j++)
                sum += Math.Abs(m[i, j]);
            if (sum > etalon)
                etalon = sum;
        }
        return etalon;
    }

    static double MinOwn(double[,] m, double[] ans, int N, int it_cnt)
    {
        double prev = 0, next = 0;
        double[] temp = new double[101];
        double[,] b = new double[101, 101];
        double A = Norm(m, N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                b[i, j] = (i == j) ? A - m[i, j] : -m[i, j];

        for (int i = 1; i <= N; i++)
            ans[i] = 1;

        for (int i = 1; i <= it_cnt; i++)
        {
            MultiplyString(b, ans, temp, N);
            prev = ans.Sum(x => x * x);
            next = ans.Zip(temp, (x, y) => x * y).Sum();
            RewriteString(temp, ans, N);
        }
        return A - (next / prev);
    }

    static void YacobiRotate(double[,] m, double[] ans, int N, int it_cnt)
    {
        if (!IsSymmetric(m, N))
        {
            Console.WriteLine("Error: the array is not symmetric. This method works only for symmetric arrays");
            return;
        }

        for (int i = 1; i <= it_cnt; i++)
        {
            MaxSearch(m, N, out int imax, out int jmax);
            double h = 2 * m[imax, jmax] / (m[imax, imax] - m[jmax, jmax]);
            double phi = Math.Atan(h) / 2;
            double[,] u = new double[101, 101];
            double[,] u_t = new double[101, 101];
            Reset(u, u_t, N);

            for (int j = 1; j <= N; j++)
            {
                u[j, j] = 1;
                u_t[j, j] = 1;
            }

            u[imax, imax] = u_t[imax, imax] = Math.Cos(phi);
            u[imax, jmax] = Math.Sin(phi);
            u_t[jmax, imax] = Math.Sin(phi);
            u[jmax, imax] = -Math.Sin(phi);
            u_t[imax, jmax] = -Math.Sin(phi);
            u[jmax, jmax] = u_t[jmax, jmax] = Math.Cos(phi);

            double[,] temp = new double[101, 101];
            double[,] temp2 = new double[101, 101];
            MultiplySquare(u, m, temp, N);
            MultiplySquare(temp, u_t, temp2, N);
            RewriteSquare(temp2, m, N);
        }

        for (int i = 1; i <= N; i++)
            ans[i] = m[i, i];
    }

    static void Main()
    {
        Console.WriteLine("This program is finding the own numbers of the square matrix using the Yacobi rotation method");
        Console.WriteLine("It also finds the smallest own number using the power method");
        Console.WriteLine("Please enter the size and the values of your matrix");
        int N = int.Parse(Console.ReadLine());
        double[,] m1 = new double[101, 101];
        double[,] m2 = new double[101, 101];
        double[] ans = new double[101];
        ReadMatrix(m1, N);
        RewriteSquare(m1, m2, N);
        Console.WriteLine("Now enter the number of iterations you want to make");
        int it_cnt = int.Parse(Console.ReadLine());

        YacobiRotate(m1, ans, N, it_cnt);
        Console.WriteLine("The own numbers(calculated using the Yacobi rotation method) of your matrix are:");
        for (int i = 1; i <= N; i++)
            Console.WriteLine($"l{i}={ans[i]}");

        Console.WriteLine("The smallest own number(calculated using the power method) of your matrix is: " + MinOwn(m2, ans, N, it_cnt));





        Console.WriteLine("This program is solving the system of non-linear equations:");
        Console.WriteLine("sin(2x - y) - 1.2x = 0.4 and 0.8x^2 + 1.5y^2 = 1 using the Modified Newton method.");
        Console.WriteLine("Please enter your initial approximations of x and y:");
        double x = double.Parse(Console.ReadLine());
        double y = double.Parse(Console.ReadLine());

        Console.WriteLine("Please enter the number of iterations:");
        int maxIterations = int.Parse(Console.ReadLine());

        RunModifiedNewtonMethod(x, y, maxIterations);
    }
}
