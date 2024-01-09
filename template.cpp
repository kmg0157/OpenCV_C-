#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#define GetMax(x,y) ((x>y)?x:y)
#define GetMin(x,y)	((x<y)?x:y)
#include <opencv2/opencv.hpp>

using namespace cv;

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}


float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0        Ե        

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B
		+ (1.0 - dx) * dy * C + dx * dy * D;

	return((int)(value + 0.5));
}


void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}

//1장. 픽셀 값 다루기

int Ex1_1()	//128x256 영상의 모든 픽셀값을 128의 밝기값으로 설정
{
	int height = 128, width = 256;
	int** img = (int**)IntAlloc2(height, width);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img[y][x] = 128;
		}
	}

	ImageShow((char*)"output", img, height, width);

	return 0;
}

int Ex1_2()	//이미지 읽기
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	ImageShow((char*)"output", img, height, width);
	return 0;
}

int Ex1_3()	//영상 이진화, 밝기값이 128이상인 픽셀은 255으로, 아니면 0으로 이진화
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] >= 128)
			{
				img_out[y][x] = 255;
			}
			else
				img_out[y][x] = 0;
		}
	}

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

void Thresholding(int threshold, int** img, int height, int width, int** img_out)	//이진화 함수
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] >= threshold)
			{
				img_out[y][x] = 255;
			}
			else
				img_out[y][x] = 0;
		}
	}
}

int Ex1_4()	//함수를 사용하여 이진화, 50, 100, 150, 200일때 출력
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	for (int threshold = 50; threshold <= 200; threshold += 50)
	{
		Thresholding(threshold, img, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
	}

	return 0;
}

int Ex1_5()	//범위마다 다른 threshold로 이진화
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	for (int y = 0; y < height / 2; y++)
	{
		for (int x = 0; x < width / 2; x++)
		{
			if (img[y][x] > 50)
			{
				img_out[y][x] = 255;
			}
			else
				img_out[y][x] = 0;

		}
	}
	for (int y = 0; y < height / 2; y++)
	{
		for (int x = width / 2; x < width; x++)
		{
			if (img[y][x] > 100)
			{
				img_out[y][x] = 255;
			}
			else
				img_out[y][x] = 0;

		}
	}
	for (int y = height / 2; y < height; y++)
	{
		for (int x = 0; x < width / 2; x++)
		{
			if (img[y][x] > 150)
			{
				img_out[y][x] = 255;
			}
			else
				img_out[y][x] = 0;

		}
	}for (int y = height / 2; y < height; y++)
	{
		for (int x = width / 2; x < width; x++)
		{
			if (img[y][x] > 200)
			{
				img_out[y][x] = 255;
			}
			else
				img_out[y][x] = 0;
		}
	}
	ImageShow((char*)"output", img_out, height, width);
	return 0;
}



//2장. 클리핑과 영상 혼합
//클리핑: 범위를 벗어나는 값을 범위내로 조정
/*
산술연산의 일종
가중치에 따라 각 입력영상의 반영 정도가 결정
동영상의 경우, fade-in fade-out 효과
*/

void AddValue2Image(int value, int** img, int height, int width, int** img_out)	//밝기값 표현 범위 넘어가는 이미지 만드는 함수		
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = img[y][x] + value;
		}
	}
}

void ImageClipping(int** img, int height, int width, int** img_out)		// 클리핑 함수
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] > 255)	//값이 255보다 크면
			{
				img_out[y][x] = 255;
			}
			else if (img[y][x] < 0)		//값이 0보다 작으면
			{
				img_out[y][x] = 0;
			}
			else	//이외의 경우
			{
				img_out[y][x] = img[y][x];
			}
		}
	}
}

int Ex2_1()	//클리핑: 픽셀값이 일정 범위를 넘어가는 경우, 표현 범위내로 값을 변환
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	AddValue2Image(50, img, height, width, img);

	ImageClipping(img, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

void ClippingMacro(int** img, int height, int width, int** img_out)	//클리핑을 매크로를 활용
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = GetMin(img[y][x], 255);
			img_out[y][x] = GetMax(img[y][x], 0);
			//한줄로도 표현 가능
			//img_out[y][x]=GetMin(GetMax(img[y][x],0),255);
		}
	}
}

//영상혼합: 두 이미지를 적절한 가중치를 주어 합하는 것
void ImageMixing(float alpha, int** img1, int** img2, int height, int width, int** img_out)	//영상혼합 함수
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = alpha * img1[y][x] + (1.0 - alpha) * img2[y][x];	//alpha는 1보다 작아야함
		}
	}
}

int Ex2_2()	//2개의 영상을 읽어 1/2씩 혼합
{
	int height, width;
	int** img1 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	float alpha = 0.5;	
	ImageMixing(alpha, img1, img2, height, width, img_out);

	ImageShow((char*)"input1", img1, height, width);
	ImageShow((char*)"input2", img2, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

int Ex2_3()	//alpha=0.1, 0.2, ..., 0.8씩 영상혼합
{
	int height, width;
	int** img1 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	for (float alpha = 0.1; alpha < 0.9; alpha += 0.1)
	{
		ImageMixing(alpha, img1, img2, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
	}

	return 0;
}





//3장.스트레칭과 히스토그램
//스트레칭: 픽셀의 밝기 값을 다른 밝기 값으로 변환하는 것
//명암대비를 위해 사용 가능

void ImageStretch(int a, int b, int c, int d, int** img, int height, int width, int** img_out)	//스트레칭 함수
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] < a)	//범위와 식은 그림에 맞춰서 알아서 바꾸자!
			{
				img_out[y][x] = (int)((float)c / a * img[y][x] + 0.5);	//0.5더하고 int로 캐스트 변환을 통해 반올림
			}
			else if (img[y][x] < b)
			{
				img_out[y][x] = (int)(((float)d - c) / (b - a) * (img[y][x] - a) + c + 0.5);
			}
			else
				img_out[y][x] = (int)((255.0 - d) / (255.0 - b) * (img[y][x] - b) + d + 0.5);
		}
	}
}

int Ex3_1()	
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int a = 100, b = 150, c = 50, d = 200;
	ImageStretch(a, b, c, d, img, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

#define RoundUp(x) ((int)(x+0.5))	//반올림 매크로

int Ex3_2()	//반올림 예제
{
	int y = RoundUp(2.6);
	printf("%d", y);

	return 0;
}



//히스토그램: 밝기 값의 빈도 수
/*
어두운영상: 밝기값이 낮은 쪽에 빈도 높음, 밝은 영상: 밝기값이 높은 쪽에 빈도 높음
낮은 명암대비: 밝기값이 한 쪽게 몰림, 높은 명암 대비: 골고루 밝기값 분산
*/
void Histogram(int** img, int height, int width, int* Hist)	//히스토그램을 저장할 배열선언, 배열에 값 저장
{
	for (int i = 0; i < 256; i++)
	{
		Hist[i] = 0;
	}
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Hist[img[y][x]]++;
		}
	}

}

int Ex3_3()	//히스토그램 출력
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);

	int Hist[256] = { 0 };
	Histogram(img, height, width, Hist);

	ImageShow((char*)"input", img, height, width);
	DrawHistogram((char*)"히스토그램", Hist);

	return 0;
}

void C_Histogram(int** img, int height, int width, int* C_Hist)	//누적 히스토그램: 히스토그램의 적분
{
	int Hist[256] = { 0 };
	Histogram(img, height, width, Hist);

	for (int i = 0; i < 256; i++)
	{
		C_Hist[i] = 0;
	}

	C_Hist[0] = Hist[0];
	for (int i = 1; i < 256; i++)
	{
		C_Hist[i] = Hist[i] + C_Hist[i - 1];
	}
}

int Ex3_4()	//누적 히스토그램 출력
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);

	int C_Hist[256] = { 0 };
	C_Histogram(img, height, width, C_Hist);

	ImageShow((char*)"input", img, height, width);
	DrawHistogram((char*)"누적 히스토그램", C_Hist);

	return 0;
}


void Norm_C_Histogram(int** img, int height, int width, int* NC_Hist)	//정규화된 누적히스토그램 함수
{
	int C_Hist[256] = { 0 };
	C_Histogram(img, height, width, C_Hist);

	int N = height * width;
	for (int i = 0; i < 256; i++)
	{
		NC_Hist[i] = C_Hist[i] * 255.0 / N;	//이게 정규화 시키는 공식
	}
}

int Ex3_5()	//정규화된 누적 히스토그램 출력
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);

	int NC_Hist[256] = { 0 };
	Norm_C_Histogram(img, height, width, NC_Hist);

	ImageShow((char*)"input", img, height, width);
	DrawHistogram((char*)"정규화된 누적 히스토그램", NC_Hist);

	return 0;
}

//히스토그램 평활화 함수
//정규화된 누적 히스토그램을 함수로 하여 스트레칭 하는 것을 히스토그램 평활화
//히스토그램 분포가 균일 분포가 됨
//히스토그램이 특정 밝기 값에 집중된 경우 효과적임
void HistogramEqualization(int** img, int height, int width, int** img_out)	
{
	int NC_Hist[256] = { 0 };
	Norm_C_Histogram(img, height, width, NC_Hist);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = NC_Hist[img[y][x]];
		}
	}
}

int Ex3_6()	//히스토그램 평활화 결과 출력
{
	//히스토그램 평활화 과정= 히스토그램->누적 히스토그램->정규화된 히스토그램->평활화
	int height, width;
	int** img = (int**)ReadImage((char*)"lenax0.5.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	HistogramEqualization(img, height, width, img_out);	//히스토그램 평활화 수행

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}




//4장. 평균 필터

//평균필터(3x3): 중심픽셀 밝기값과 주변의 8개 픽셀 밝기값의 평균값을 출력
/*
	img[y-1][x-1]	img[y-1][x]		img[y-1][x+1]
	img[y][x-1]		img[y][x]		img[y][x+1]
	img[y+1][x-1]	img[y+1][x]		img[y+1][x+1]
*/
void Avg3x3(int** img, int height, int width, int** img_out)	//3x3 평균필터 함수
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1)	//가장자리 처리: 입력 값 그대로 적용
			{
				img_out[y][x] == img[y][x];
			}
			else	//가장자리가 아닌 경우
			{
				img_out[y][x] = (int)((img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1]
					+ img[y][x - 1] + img[y][x] + img[y][x + 1]
					+ img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1]) / 9.0 + 0.5);	//평균값 구한 후 반올림 적용
			}
		}
	}

	//반복문을 사용한 가장자리 처리
	/*
	for (int x = 0; x < width; x++)
	{
		img_out[0][x] = img[0][x];
		img_out[height - 1][x] = img[height - 1][x];
	}

	for (int y = 0; y < height; y++)
	{
		img_out[y][0] = img[y][0];
		img_out[y][width - 1] = img[y][width - 1];
	}

	//가장자리가 아닌 경우
	for (int y = 1; y < height-1; y++)
	{
		for (int x = 1; x < width-1; x++)
		{
			img_out[y][x] = (int)((img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1]
				+ img[y][x - 1] + img[y][x] + img[y][x + 1]
				+ img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1])
				/ 9.0 + 0.5);
		}
	}
	*/
}

int Ex4_1()	//3x3 평균필터 적용 결과 출력
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	Avg3x3(img, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

//NxN 평균필터 함수
//N이 커질수록 점점 흐릿해짐
void AvgNxN(int N, int** img, int height, int width, int** img_out)	
{
	int delta = (N - 1) / 2; // delta : 가장자리 폭, N : 마스크 크기
	// 가장자리 처리
	for (int y = 0; y < delta; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = img[y][x]; // 최상단 가로선
			img_out[height - 1 - y][x] = img[height - 1 - y][x]; // 최하단 가로선
		}
	}
	for (int x = 0; x < delta; x++) {
		for (int y = 0; y < height; y++) {
			img_out[y][x] = img[y][x]; // 최좌측 세로선
			img_out[y][width - 1 - x] = img[y][width - 1 - x]; // 최우측 세로선
		}
	}
	//가장자리가 아닐 경우
	for (int y = delta; y < height - delta; y++) {
		for (int x = delta; x < width - delta; x++) {
			img_out[y][x] = 0;
			for (int cy = -delta; cy <= delta; cy++) {
				for (int cx = -delta; cx <= delta; cx++) {
					img_out[y][x] += img[y + cy][x + cx];
				}
			}
			img_out[y][x] = (int)((float)img_out[y][x] / (N * N) + 0.5);
		}
	}
}

int Ex4_2()	//평균필터(nxn)적용
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	for (int N = 3; N <= 15; N++)
	{
		AvgNxN(N, img, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
	}
	return 0;
}

//가장자리 처리: 영상 바깥쪽 픽셀의 밝기값 처리
//가장자리를 0으로 처리
void AvgNxN_BoundaryZero(int N, int** img, int height, int width, int** img_out)    
{
	int delta = (N - 1) / 2; // 가장자리 폭
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = 0;
			for (int cy = -delta; cy <= delta; cy++) {
				for (int cx = -delta; cx <= delta; cx++) {
					if (!(y + cy < 0 || y + cy >= height || x + cx < 0 || x + cx >= width))
						img_out[y][x] += img[y + cy][x + cx];
				}
			}
			img_out[y][x] = (int)((float)img_out[y][x] / (N * N) + 0.5);
		}
	}
}

//가장자리를 가장 가까운 픽셀 밝기값으로 대체
//이질적인 현상을 줄이는 가장 자연스러운 방법
void AvgNxN_BoundaryNearestPixel(int N, int** img, int height, int width, int** img_out) 
{
	int delta = (N - 1) / 2; // 가장자리 폭
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = 0;
			for (int cy = -delta; cy <= delta; cy++) {
				for (int cx = -delta; cx <= delta; cx++) {
					int pos_y = GetMax(GetMin(y + cy, height - 1), 0);
					int pos_x = GetMax(GetMin(x + cx, width - 1), 0);
					img_out[y][x] += img[pos_y][pos_x];
				}
			}
			img_out[y][x] = (int)((float)img_out[y][x] / (N * N) + 0.5);
		}
	}
}
int Ex4_3()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	AvgNxN_BoundaryZero(9, img, height, width, img_out1);
	AvgNxN_BoundaryNearestPixel(9, img, height, width, img_out2);
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기1", img_out1, height, width);
	ImageShow((char*)"출력영상보기2", img_out2, height, width);

	return 0;
}



//5장. 마스킹과 콘볼루션
//마스킹: 평균필터 연산의 새로운 관점 -> 마스크와 해당 영역의 곱(계수값을 자유롭게 변경 가능)
//영상을 흐릿하게 만드는 효과, 영상 내의 잡음 감소 효과
void Avg3x3_WithMask(int** img, int height, int width, int** img_out)
{
	//마스크
	float mask[3][3] = { 1.0 / 9, 1.0 / 9, 1.0 / 9,
						1.0 / 9, 1.0 / 9, 1.0 / 9,
						1.0 / 9, 1.0 / 9, 1.0 / 9 };
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1)	//가장자리 처리
			{
				img_out[y][x] == img[y][x];
			}
			else	//가장자리가 아닌 경우
			{
				float output = 0.0;

				for (int cy = -1; cy <= 1; cy++)
				{
					for (int cx = -1; cx <= 1; cx++)
					{
						output += mask[cy + 1][cx + 1] * img[y + cy][x + cx];	//mask[1][1]이 중앙 픽셀에 곱해져야하므로
					}
				}

				img_out[y][x] = (int)(output + 0.5);
			}
		}
	}
}

int Ex5_1()	//마스크를 이용한 3x3 평균 필터
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	Avg3x3_WithMask(img, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

//마스크를 입력 가능한 함수
void Avg3x3_WithMaskInput(float** mask, int** img, int height, int width, int** img_out)	
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1)	//가장자리 처리
			{
				img_out[y][x] == img[y][x];
			}
			else	//가장자리가 아닌 경우
			{
				float output = 0.0;

				for (int cy = -1; cy <= 1; cy++)
				{
					for (int cx = -1; cx <= 1; cx++)
					{
						output += mask[cy + 1][cx + 1] * img[y + cy][x + cx];
					}
				}

				img_out[y][x] = (int)(output + 0.5);
			}
		}
	}
}

int Ex5_2()	//마스크 입력함수 사용
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	float** mask = (float**)FloatAlloc2(3, 3);
	mask[0][0] = 1.0 / 9; mask[0][1] = 1.0 / 9; mask[0][2] = 1.0 / 9;
	mask[1][0] = 1.0 / 9; mask[1][1] = 1.0 / 9; mask[1][2] = 1.0 / 9;
	mask[2][0] = 1.0 / 9; mask[2][1] = 1.0 / 9; mask[2][2] = 1.0 / 9;

	Avg3x3_WithMaskInput(mask, img, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

//NxN 마스크 입력 가능 함수
void AvgNxN_BoundaryNearestPixel_WithMaskInput(float** mask, int N, int** img, int height, int width, int** img_out)	
{
	int delta = (N - 1) / 2; // 가장자리 폭
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float output = 0.0;
			for (int cy = -delta; cy <= delta; cy++) {
				for (int cx = -delta; cx <= delta; cx++) {
					int pos_y = GetMax(GetMin(y + cy, height - 1), 0);
					int pos_x = GetMax(GetMin(x + cx, width - 1), 0);
					output += mask[cy + delta][cx + delta] * img[pos_y][pos_x];
				}
			}
			img_out[y][x] = (int)(output + 0.5);
		}
	}
}
int Ex5_3()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbaraGN15.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int N = 9;	//마스크 크기
	float** mask = (float**)FloatAlloc2(N, N);
	for (int cy = 0; cy < N; cy++)
		for (int cx = 0; cx < N; cx++)
			mask[cy][cx] = 1.0 / (N * N);

	AvgNxN_BoundaryNearestPixel_WithMaskInput(mask, N, img, height, width, img_out);
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}




//6장. 에지 검출
//에지검출 연산자
//	미분연산자->잡음에 민감
//	라플라시안 연산자: 방향성이 없음->잡음에 민감
//	소벨, 로버트, 프리윗 연산자->잡음에 상대적으로 덜 민감
//미분 연산자: f'(x)=f(x+1)-f(x)
void MagGradient(int** img, int height, int width, int** img_out)	//그래디언트 크기 구하기: |fx|+|fy|
{
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int fx = img[y][x + 1] - img[y][x]; // x방향 기울기
			int fy = img[y + 1][x] - img[y][x]; // y방향 기울기
			img_out[y][x] = abs(fx) + abs(fy); // 그라디언트의 크기
		}
	}
}
void MagGradient_X(int** img, int height, int width, int** img_out)	//x방향 그래디언트 크기: fx=f(x+1,y)-f(x,y)
{
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			img_out[y][x] = abs(img[y][x + 1] - img[y][x]); // x방향 기울기 크기
		}
	}
}
void MagGradient_Y(int** img, int height, int width, int** img_out)	//y방향 그래디언트 크기: fy=f(x,y+1)-f(x,y)
{
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			img_out[y][x] = abs(img[y + 1][x] - img[y][x]); // y방향 기울기 크기
		}
	}
}

int FindMaxValue(int** img, int height, int width)	//최대값 찾는 함수
{
	int max_value = img[0][0]; // 초기값
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] > max_value) // 최대값을 구하는 부분
				max_value = img[y][x];
		}
	}
	return(max_value);
}

void NormalizeByMax(int** img, int height, int width)	//정규화 함수
{
	// 정규화를 위해 최대값 찾기
	int max_value = FindMaxValue(img, height, width);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img[y][x] = 255 * img[y][x] / max_value;
		}
	}
}
int Ex6_1()	//미분 크기의 정규화(0~255로 스케일링)
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lenaGN15.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	MagGradient(img, height, width, img_out); // 그라디언트 크기를 img_out에 씀
	NormalizeByMax(img_out, height, width); // img_out을 0~255 내로정규화

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

//각 방향 미분 결과 출력
int Ex6_2()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	MagGradient_X(img, height, width, img_out1); // 그라디언트 크기를 img_out에 씀
	MagGradient_Y(img, height, width, img_out2);
	NormalizeByMax(img_out1, height, width); // img_out을 0~255 내로정규화
	NormalizeByMax(img_out2, height, width);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out1, height, width);
	ImageShow((char*)"출력영상보기", img_out2, height, width);

	return 0;
}

//라플라시안 연산자=2차 미분 연산자
//방향이 없고 크기만 존재, 마스킹 함수 재활용

void AbsImage(int** img, int height, int width)		//라플라시안 크기 구하는 함수
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img[y][x] = abs(img[y][x]);
		}
	}
}

int Ex6_3()	//라플라시안 사용
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	int N = 3;
	float** mask = (float**)FloatAlloc2(N, N);
	
	mask[0][0] = -1; mask[0][1] = -1; mask[0][2] = -1;
	mask[1][0] = -1; mask[1][1] = 8; mask[1][2] = -1;
	mask[2][0] = -1; mask[2][1] = -1; mask[2][2] = -1;
	
	AvgNxN_BoundaryNearestPixel_WithMaskInput(mask, N, img, height, width, img_out);	//마스킹 콘볼루션 함수
	AbsImage(img_out, height, width); // 절대치를 취함
	NormalizeByMax(img_out, height, width); // img_out을 0~255 내로정규화
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

//기타 연산자
// 미분 연산자는 바로 이웃한 픽셀 사이의 차이를 계산
// 3개 쌍의 합으로 계산 -> 평균 효과가 있음 -> 잡음에 강함

/* 소벨
-1 -2 -1		1	0	-1
 0  0  0	->  2	0	-2
 1  2  1		1	0	-1
*/

//소벨 크기 구하기
void MagSobel(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int fx = (img[y - 1][x + 1] - img[y - 1][x - 1]) + 2 * (img[y][x + 1] - img[y][x - 1]) + (img[y + 1][x + 1] - img[y + 1][x - 1]);
			int fy = (img[y + 1][x - 1] - img[y - 1][x - 1]) + 2 * (img[y + 1][x] - img[y - 1][x]) + (img[y + 1][x + 1] - img[y - 1][x + 1]);
			img_out[y][x] = abs(fx) + abs(fy); // 그라디언트의 크기
		}
	}
}

//x방향 소벨 연산 크기
void MagSobel_X(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int fx = (img[y - 1][x + 1] - img[y - 1][x - 1]) + 2 * (img[y][x + 1] - img[y][x - 1]) + (img[y + 1][x + 1] - img[y + 1][x - 1]);
			img_out[y][x] = abs(fx); // x방향 기울기
		}
	}
}

//y방향 소벨 연산 크기

void MagSobel_Y(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int fy = (img[y + 1][x - 1] - img[y - 1][x - 1]) + 2 * (img[y + 1][x] - img[y - 1][x]) + (img[y + 1][x + 1] - img[y - 1][x + 1]);
			img_out[y][x] = abs(fy); // y방향 기울기
		}
	}
}

int Ex6_4()	//소벨 연산자, 소벨 크기 구하기
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lenaGN15.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	MagSobel(img, height, width, img_out); // 소벨 연산 결과를 img_out에 씀
	NormalizeByMax(img_out, height, width); // img_out을 0~255 내로정규화

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

int Ex6_5()	//x방향 및 y방향 소벨 연산
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out_x = (int**)IntAlloc2(height, width);
	int** img_out_y = (int**)IntAlloc2(height, width);

	MagSobel_X(img, height, width, img_out_x);
	MagSobel_Y(img, height, width, img_out_y);
	NormalizeByMax(img_out_x, height, width);
	NormalizeByMax(img_out_y, height, width);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력(fx)영상보기", img_out_x, height, width);
	ImageShow((char*)"출력(fy)영상보기", img_out_y, height, width);

	return 0;
}

//마스크 계수 입력이 가능한 소벨 연산
int Ex6_6()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	int N = 3;
	float** mask = (float**)FloatAlloc2(N, N);
	mask[0][0] = 1; mask[0][1] = 0; mask[0][2] = -1;
	mask[1][0] = 2; mask[1][1] = 0; mask[1][2] = -2;
	mask[2][0] = 1; mask[2][1] = 0; mask[2][2] = -1;
	
	AvgNxN_BoundaryNearestPixel_WithMaskInput(mask, N, img, height, width, img_out);
	AbsImage(img_out, height, width); // 절대치를 취함
	NormalizeByMax(img_out, height, width); // img_out을 0~255 내로정규화
	
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}



//7장. 선명화 처리 및 중간값 필터
//경계부분의 명압대비를 증폭시킴: 밝은 부분은 더 밝게, 어두운 부분은 더 어둡게
//잡음이 있음-> 잡음도 강화됨
//잡음이 없음-> 또렷한 영상, 강도가 과하면 잡음이 존재하는 것과 같은 결과

// 선명화 마스크: 입력영상과 에지검출결과를 조합한 결과를 출력
//선명화 처리 마스크 계수의 합은 1이어야함
/*예시) 라플라시안 에지 검출이 많이 사용됨
-1	-1	-1		0	0	0		-1	-1	-1
-1	 8	-1	+	0	1	0	=	-1	 9	-1
-1	-1	-1		0	0	0		-1	-1	-1
*/

int Ex7_1()	//선명화처리 마스크
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lenaGN15.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	int N = 3;
	float** mask = (float**)FloatAlloc2(N, N);
	mask[0][0] = -1; mask[0][1] = -1; mask[0][2] = -1;
	mask[1][0] = -1; mask[1][1] = 9; mask[1][2] = -1;
	mask[2][0] = -1; mask[2][1] = -1; mask[2][2] = -1;
	
	AvgNxN_BoundaryNearestPixel_WithMaskInput(mask, N, img, height, width, img_out);	//마스크 처리
	ImageClipping(img_out, height, width, img_out); // 음수 및 255 초과 값을 범위내로 클리핑
	
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

//선명화 강도 조절 구현
//에지 검출 시 가중치 곱함, 가중치는 1 이하의 값
//가중치를 조절하여도 최종적인 선명화 처리 마스크 계수의 합은 1이어야함
int Ex7_2()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int N = 3;
	float alpha = 0.5; // 가중치, 클수록 에지가 더 강조됨
	float** mask = (float**)FloatAlloc2(N, N);
	mask[0][0] = -alpha; mask[0][1] = -alpha; mask[0][2] = -alpha;
	mask[1][0] = -alpha; mask[1][1] = 1 + 8.0 * alpha; mask[1][2] = -alpha;
	mask[2][0] = -alpha; mask[2][1] = -alpha; mask[2][2] = -alpha;
	
	AvgNxN_BoundaryNearestPixel_WithMaskInput(mask, N, img, height, width, img_out);
	ImageClipping(img_out, height, width, img_out); // 음수 및 255 초과 값을 범위내로 클리핑
	
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

//중간값 필터: 버블소팅 후 중간 순위에 있는 값 선택
//순서: 픽셀 값 읽기 -> 정렬 -> 중간값 선택 -> 출력 영상에 반영
//salt and pepper 잡음이 낀 영상에 효과적
/*
버블정렬
1. N개 중 최대값을 가장 아래로 위치
2. N-1 개 중 최대값을 가장 아래로 위치
3. N-i 개 ...반복
*/
void Swap(int* a, int* b)
{
	int buff = *a;
	*a = *b; *b = buff;
}
void Bubbling(int* A, int num)	//이웃 값과 위치 바꾸기
{
	for (int i = 0; i < num - 1; i++) {
		if (A[i] > A[i + 1]) Swap(&A[i], &A[i + 1]); // 바로 이웃한 값끼리 위치 바꾸기
	}
}
void BubbleSort(int* A, int N)	//버블링 함수 반복
{
	for (int i = 0; i < N - 1; i++) // 버블링 반복, 맨처음은 N개에 대해, 두번째는 (N-1)개에 대해
		Bubbling(A, N - i);
}

//2차원 배열을 1차원 배열에 복사
//마스크 내에 있는 픽셀 값을 1차언 배열로 만들어 입력시켜야함
/*
1. 2D->1D
for(int y=0;y<3;y++)
{
	for(int x=0;x<3;x++)
	{
		data1D[y*3+x]=data2D[y][x];
	}
}
2. 1D->2D
for(int y=0;y<3;y++)
{
	for(int x=0;x<3;x++)
	{
		data2D[y][x]=data1D[y*3+x];
	}
}
*/

//가장자리 예외처리(가장 가까운 픽셀값으로 대체)
void GetBlock1D(int y0, int x0, int dy, int dx, int** img, int height, int width, int* data1D)	
{
	for (int y = 0; y < dy; y++) {
		for (int x = 0; x < dx; x++) {
			int pos_y = GetMax(GetMin(y + y0, height - 1), 0);
			int pos_x = GetMax(GetMin(x + x0, width - 1), 0);
			data1D[y * dx + x] = img[pos_y][pos_x];
		}
	}
}

//중간값 필터
void MedianFiltering(int Num, int** img, int height, int width, int** img_out)
{
	int h_Num = (Num - 1) / 2;
	int* data1D = (int*)calloc(Num * Num, sizeof(int)); // 메모리 할당
	int median_index = (Num * Num - 1) / 2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			GetBlock1D(y - h_Num, x - h_Num, Num, Num, img, height, width, data1D);		//현재 좌표를 마스크 중심에 가도록 설정
			BubbleSort(data1D, Num * Num);
			img_out[y][x] = data1D[median_index];
		}
	}
	free(data1D); // 메모리 해제
}
int Ex7_3()	//중간값 필터 적용
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lenaSP20.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	MedianFiltering(3, img, height, width, img_out1);
	MedianFiltering(3, img_out1, height, width, img_out2);	//중간값 필터 2번 적용

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"중간값 필터 1번", img_out1, height, width);
	ImageShow((char*)"중간값 필터 2번", img_out2, height, width);

	return 0;
}

//중간값 필터 k번 적용
//잡음 제거 효과는 증가하나 영상에 흐림(blur)가 발생
int Ex7_4()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lenaSP20.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int MaskSize = 3; // 마스크 크기
	int K = 10; // 반복 적용 횟수

	MedianFiltering(MaskSize, img, height, width, img_out);
	for (int i = 0; i < K - 1; i++) {
		MedianFiltering(MaskSize, img_out, height, width, img_out);
	}
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력(K번적용)영상보기", img_out, height, width);

	return 0;
}



//8장. 크기 변환과 쌍선형 보간

//upsampling: 단순 2배 확대(주변 픽셀 단순 복사하여 빈자리 채우기)
void UpSamplingx2_repeat(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[2 * y][2 * x] = img[y][x];
			img_out[2 * y][2 * x + 1] = img[y][x];
			img_out[2 * y + 1][2 * x] = img[y][x];
			img_out[2 * y + 1][2 * x + 1] = img[y][x];
		}
	}
}
int Ex8_1()	//upsampling
{
	int height, width;
	int** img = (int**)ReadImage((char*)"s_lena.png", &height, &width);
	int** img_out = (int**)(int**)IntAlloc2(2 * height, 2 * width);
	UpSamplingx2_repeat(img, height, width, img_out);
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, 2 * height, 2 * width);

	return 0;
}

//빈공간을 주변 픽셀의 평균으로 채우는 upsampling
/*
A	E	B			E=(A+B)/2	
F	I	G			F=(A+C)/2	I=(A+B+C+D)/4	G=(B+D)/2
C	H	D			H=(C+D)/2
*/

void UpSamplingx2_avg(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int A = img[y][x];
			int B = img[y][GetMin(x + 1, width - 1)];
			int C = img[GetMin(y + 1, height - 1)][x];
			int D = img[GetMin(y + 1, height - 1)][GetMin(x + 1, width - 1)];
			img_out[2 * y][2 * x] = img[y][x]; // A
			img_out[2 * y][2 * x + 1] = (A + B + 1) / 2; // E		반올림처리
			img_out[2 * y + 1][2 * x] = (A + C + 1) / 2; // F		반올림처리
			img_out[2 * y + 1][2 * x + 1] = (A + B + C + D + 2) / 4; // I		반올림처리
		}
	}
}
int Ex8_2()	//upsampling_avg
{
	int height, width;
	int** img = (int**)ReadImage((char*)"s_lena.png", &height, &width);
	int** img_out = (int**)(int**)IntAlloc2(2 * height, 2 * width);
	UpSamplingx2_avg(img, height, width, img_out);
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, 2 * height, 2 * width);

	return 0;
}

// downsampling: 1/2로 축소
void DownSamplingx2(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y += 2) {
		for (int x = 0; x < width; x += 2) {
			img_out[y / 2][x / 2] = img[y][x];
		}
	}
}

int Ex8_3()	//downsampling: (x,y)좌표가 짝수인 위치 픽셀만 모아 1/2로 줄임
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)(int**)IntAlloc2(height / 2, width / 2);
	
	DownSamplingx2(img, height, width, img_out);
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height / 2, width / 2);

	return 0;
}

//쌍선형 보간: 직선상의 밝기로 채움(2차원 신호 대상)
// 선형보간을 x 방향과 y 방향에 대해 수행
//2의 승수 배 확대 및 축소가 아닌 경우
//0~1이 100~200이라면 0.25=125, 0.5=150, 0.75=175 -> 100과 200을 직선으로 이어 이 직선상의 밝기값 -> 선형 보간

int BilinearInterpolation(double y, double x, int** image, int height, int width)
{
	int x_int = (int)x;
	int y_int = (int)y;
	int A = image[GetMin(GetMax(y_int, 0), height - 1)][GetMin(GetMax(x_int, 0), width - 1)];
	int B = image[GetMin(GetMax(y_int, 0), height - 1)][GetMin(GetMax(x_int + 1, 0), width - 1)];
	int C = image[GetMin(GetMax(y_int + 1, 0), height - 1)][GetMin(GetMax(x_int, 0), width - 1)];
	int D = image[GetMin(GetMax(y_int + 1, 0), height - 1)][GetMin(GetMax(x_int + 1, 0), width - 1)];
	double dx = x - x_int;
	double dy = y - y_int;
	double value = (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B + (1.0 - dx) * dy * C + dx * dy * D;
	return((int)(value + 0.5));
}
int Ex8_4()	//쌍선형보간을 이용해 2배 확대
{
	int height, width;
	int** img = (int**)ReadImage((char*)"s_lena.png", &height, &width);
	int** img_out = (int**)(int**)IntAlloc2(2 * height, 2 * width);
	for (int y = 0; y < 2 * height; y++) {
		for (int x = 0; x < 2 * width; x++) {
			img_out[y][x] = BilinearInterpolation(y / 2.0, x / 2.0, img, height, width);
		}
	}
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, 2 * height, 2 * width);

	return 0;
}
int Ex8_5()	//쌍선형보간을 이용해 알파 배 확대
{
	int height, width;
	int** img = (int**)ReadImage((char*)"s_lena.png", &height, &width);
	float alpha = 1.3;
	int height_out = (int)(alpha * height);
	int width_out = (int)(alpha * width);
	int** img_out = (int**)(int**)IntAlloc2(height_out, width_out);
	for (int y = 0; y < height_out; y++) {
		for (int x = 0; x < width_out; x++) {
			img_out[y][x] = BilinearInterpolation(y / alpha, x / alpha, img, height, width);
		}
	}
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height_out, width_out);

	return 0;
}

//입력영상의 크기가 (height, width)이고 출력영상의 크기가(height_out,width_out)이 되도록 영상 크기 변환해주는 함수
void ResizeUsingBilinear(int** img, int height, int width, int** img_out, int height_out, int width_out)
{
	double alpha_x = (double)width_out / width;
	double alpha_y = (double)height_out / height;
	for (int y = 0; y < height_out; y++) {
		for (int x = 0; x < width_out; x++) {
			img_out[y][x] = BilinearInterpolation(y / alpha_y, x / alpha_x, img, height, width);
		}
	}
}
int Ex8_6()	//원본이미지를 내가 원하는 만큼 크기 변환
{
	int height, width;
	int** img = (int**)ReadImage((char*)"s_lena.png", &height, &width);
	
	int height_out = (int)(1.1 * height);	//x방향 배율
	int width_out = (int)(1.6 * width);		//y방향 배율
	int** img_out = (int**)IntAlloc2(height_out, width_out);
	printf("\n img_out size : h = %d, w = %d", height_out, width_out);
	
	ResizeUsingBilinear(img, height, width, img_out, height_out, width_out);
	
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height_out, width_out);

	return 0;
}



//9장. 기하학적 변환

//영상 이동: 단순히 영상을 좌우상하 방향으로 이동시키는 변환으로서 어파인 변환의 가장 간단한 형태의 변환
/*
x'=x+tx,	y'=y+ty	(tx, ty는 이동값, x', y'은 이동 후 좌표)
*/
void Translation(int ty, int tx, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = 0;	//출력 영상의 빈자리에 0으로 초기화
		}
	}
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int y_prime = y + ty;
			int x_prime = x + tx;
			//예외처리
			if (x_prime < 0 || y_prime < 0 || x_prime >= width || y_prime >= height)
				continue;	//x', y'이 영상 바깥인 경우 아무것도 수행하지 않음
			else
				img_out[y_prime][x_prime] = img[y][x];
		}
	}
}
int Ex9_1()	//영상이동
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int ty = 50, tx = 100; //x로 50만큼, y로 100만큼 이동
	Translation(ty, tx, img, height, width, img_out);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

//영상 회전: 반환 결과로 나타나는 좌표는 정수가 아닌 실수로 나타나므로, 회전 행렬의 역행렬을 이용하여 출력 영상의 좌표에 대응하는 입력 영상의 좌표를 계산하여 픽셀 값을 가져옴
/*
x=x'cos+y'sin
y=-x'sin+y'cos
*/
void Rotation(double theta, int** img, int height, int width, int** img_out)
{
	double rad = theta / 180.0 * CV_PI; // degree -> rad로 변환
	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {
			int y = -x_prime * sin(rad) + y_prime * cos(rad);
			int x = x_prime * cos(rad) + y_prime * sin(rad);
			if (x < 0 || x >= width || y < 0 || y >= height) continue;
			else img_out[y_prime][x_prime] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}
int Ex9_2()	//영상 회전: 원점(좌측 상단)을 중심으로 45도 회전
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	double theta = -15.0;	//회전 각도
	Rotation(theta, img, height, width, img_out);
	
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

//영상 중심을 축으로 회전
/*
(x0,y0): 입력영상의 회전 중심
(x'0,y'0): 출력영상의 회전 중심
x=[(x'-x'0)cos+(y'-y'0)sin]+x0
y=[-(x'-x'0)sin+(y'-y'0)cos]+y0
*/
void Rotation2(double theta, int y0, int x0, int y0_prime, int x0_prime,
	int** img, int height, int width, int** img_out)
{
	double rad = theta / 180.0 * CV_PI; // degree 　 rad로 변환
	double cos_value = cos(rad);
	double sin_value = sin(rad);
	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {
			int y = -(x_prime - x0_prime) * sin_value + (y_prime - y0_prime) * cos_value + y0;
			int x = (x_prime - x0_prime) * cos_value + (y_prime - y0_prime) * sin_value + x0;
			if (x < 0 || x >= width || y < 0 || y >= height) continue;
			else img_out[y_prime][x_prime] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}
int Ex9_3()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	double theta = -15.0;
	int y0 = height / 2, x0 = width / 2;	//입력 영상의 가운데를 회전축으로 설정
	int y0_prime = height / 2, x0_prime = width / 2;	//출력 영상의 가운데를 회전축으로 설정
	Rotation2(theta, y0, x0, y0_prime, x0_prime, img, height, width, img_out);
	
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
	
	return 0;
}

//어파인 변환: 영상 회전 시와 유사하게 어파인 변환의 역변환을 이용하여 출력 영상을 얻음
//영상 가운데를 중심으로 한 변환식
/*
x=a'(x'-tx)+b'(y'-ty)+x0
y=a'(x'-tx)+b'(y'-ty)+y0
*/
void AffineTransform(double a, double b, double c, double d,
	int y0, int x0, int y0_prime, int x0_prime,
	int** img, int height, int width, int** img_out)
{
	double a_prime = d / (a * d - b * c), b_prime = -b / (a * d - b * c);
	double c_prime = -c / (a * d - b * c), d_prime = a / (a * d - b * c);
	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {
			//어파인 변환 구현
			int y = c_prime * (x_prime - x0_prime) + d_prime * (y_prime - y0_prime) + y0;
			int x = a_prime * (x_prime - x0_prime) + b_prime * (y_prime - y0_prime) + x0;
			if (x < 0 || x >= width || y < 0 || y >= height) continue;
			else img_out[y_prime][x_prime] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}
int Ex9_4()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"s_barbara_4affine.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	double a = 1.0, b = 0.0;	//어파인 변환 계수 설정
	double c = 0.0, d = 1.0;	//어파인 변환 계수 설정
	int y0 = height / 2, x0 = width / 2;
	int y0_prime = height / 2, x0_prime = width / 2;
	AffineTransform(a, b, c, d, y0, x0, y0_prime, x0_prime, img, height, width, img_out);
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

//구조체를 이용한 어파인 함수->파라미터 간소화
struct AffineT {
	double a, b, c, d;
	int x0, y0, x0_prime, y0_prime;
};
void AffineTransform_Struct(AffineT p, int** img, int height, int width, int** img_out)
{
	double a_prime = p.d / (p.a * p.d - p.b * p.c), b_prime = -p.b / (p.a * p.d - p.b * p.c);
	double c_prime = -p.c / (p.a * p.d - p.b * p.c), d_prime = p.a / (p.a * p.d - p.b * p.c);
	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {
			int y = c_prime * (x_prime - p.x0_prime) + d_prime * (y_prime - p.y0_prime) + p.y0;
			int x = a_prime * (x_prime - p.x0_prime) + b_prime * (y_prime - p.y0_prime) + p.x0;
			if (x < 0 || x >= width || y < 0 || y >= height) continue;
			else img_out[y_prime][x_prime] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}
int Ex9_5()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"s_barbara_4affine.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	AffineT p;
	p.a = 1.0, p.b = -1.0;
	p.c = 1.0, p.d = 1.0;
	p.y0 = height / 2, p.x0 = width / 2;
	p.y0_prime = height / 2, p.x0_prime = width / 2;

	AffineTransform_Struct(p, img, height, width, img_out);
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);

	return 0;
}

//중간고사
#define PI 3.14
//선그리기
void DrawLine(int value, int y0, int x0, int y1, int x1, float thickness, int** img_out, int height, int width)
{
	float a = y1 - y0;
	float b = -(x1 - x0);
	float c = -(y1 - y0) * x0 + (x1 - x0) * y0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (y<imin(y1, y0) || y>imax(y1, y0) || x<imin(x1, x0) || x>imax(x1, x0))
				continue;
			if (fabs(a * x + b * y + c) / sqrt(a * a + b * b) <= thickness) {
				img_out[y][x] = value;
			}
		}
	}
}
void myProb1()
{
	int height = 512, width = 512;
	int** img_out = (int**)IntAlloc2(height, width);
	int x0 = 100, y0 = 100, x1 = 400, y1 = 400;
	float thickness = 20;
	DrawLine(255, y0, x0, y1, x1, thickness, img_out, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

//선 여러개 그리기
void DrawLinePlus(int value, int y0, int x0, int y1, int x1, float thickness, int** img_out, int height, int
	width)
{
	float a = y1 - y0;
	float b = -(x1 - x0);
	float c = -(y1 - y0) * x0 + (x1 - x0) * y0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (y<imin(y1, y0) || y>imax(y1, y0) || x<imin(x1, x0) || x>imax(x1, x0))
				continue;
			if (fabs(a * x + b * y + c) / sqrt(a * a + b * b) <= thickness) {
				img_out[y][x] += value;
			}
		}
	}
}
void DrawMultiLine(int* value, int* y0, int* x0, int* y1, int* x1, int num, float thickness, int** img_out,
	int height, int width)
{
	for (int i = 0; i < num; i++) {
		DrawLinePlus(value[i], y0[i], x0[i], y1[i], x1[i], thickness, img_out, height, width);
	}
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = imin(img_out[y][x], 255);
		}
	}
}
void myProb2()
{
	int height = 512, width = 512;
	int** img_out = (int**)IntAlloc2(height, width);
	int value[] = { 50, 100, 150, 200 };
	int x0[] = { 100, 200, 300, 400 };
	int y0[] = { 100, 100, 100, 100 };
	int x1[] = { 450, 350, 250, 150 };
	int y1[] = { 400, 400, 400, 400 };
	int num = 4;
	float thickness = 20;
	DrawMultiLine(value, y0, x0, y1, x1, num, thickness, img_out, height, width);
	ImageShow((char*)"output", img_out, height, width);
}
//바운딩박스
void DrawBBox(int** img, int height, int width, int** img_out)
{
	int x_max = -INT_MAX;
	int x_min = INT_MAX;
	int y_max = -INT_MAX;
	int y_min = INT_MAX;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = img[y][x];
			if (img[y][x] > 128) {
				x_max = imax(x_max, x);
				x_min = imin(x_min, x);
				y_max = imax(y_max, y);
				y_min = imin(y_min, y);
			}
		}
	}
	float thickness = 2.0;
	DrawLine(255, y_min, x_min, y_max, x_min, thickness, img_out, height, width);
	DrawLine(255, y_min, x_max, y_max, x_max, thickness, img_out, height, width);
	DrawLine(255, y_min, x_min, y_min, x_max, thickness, img_out, height, width);
	DrawLine(255, y_max, x_min, y_max, x_max, thickness, img_out, height, width);
}
void myProb3()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"test3.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	DrawBBox(img, height, width, img_out);
	ImageShow((char*)"output", img_out, height, width);
}
//필터링
void HorFiltering(float* filter, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 1; x < width - 1; x++) {
			img_out[y][x] = filter[0] * img[y][x - 1] + filter[1] * img[y][x] + filter[2] * img[y][x +
				1];
		}
	}
}
void VerFiltering(float* filter, int** img, int height, int width, int** img_out)
{
	for (int y = 1; y < height - 1; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = filter[0] * img[y - 1][x] + filter[1] * img[y][x] + filter[2] * img[y + 1][x];
		}
	}
}
void ClippingImage(int** img, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img[y][x] = imin(imax(0, img[y][x]), 255);
		}
	}
}
void myProb4()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out_x = (int**)IntAlloc2(height, width);
	int** img_out_xy = (int**)IntAlloc2(height, width);
	float filter[3] = { -0.25, 1.5, -0.25 };
	HorFiltering(filter, img, height, width, img_out_x);
	VerFiltering(filter, img_out_x, height, width, img_out_xy);
	ClippingImage(img_out_xy, height, width);
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out_xy, height, width);
}

//기말

//12장. 템플릿 매칭
//영상(블록)의 유사도 측정하기
// - 두 영상 간 차이값이 적으면 유사하다고 판단 -> MAD, MSE

float MAD(int** img1, int** img2, int height, int width)
{
	float mad = 0.0; // 초기화
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			mad += abs(img1[y][x] - img2[y][x]);
		}
	}
	return(mad / (height * width)); // 총 개수로 나눔
}
float MSE(int** img1, int** img2, int height, int width)
{
	float mse = 0.0; // 초기화
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			mse += (img1[y][x] - img2[y][x]) * (img1[y][x] - img2[y][x]);
		}
	}
	return(mse / (height * width)); // 총 개수로 나눔
}

//영상 내 일부와 템플릿 블록과의 유사도 측정
//(x+xp, y+yp)가 영상 내에 없는 경우 -> 경계처리
float MAD2(int yp, int xp, int** tplate, int dy, int dx, int** img, int height, int width)
{
	float mad = 0.0; // 초기화
	for (int y = 0; y < dy; y++) {
		for (int x = 0; x < dx; x++) {
			int pos_y = GetMin(GetMax(y + yp, 0), height - 1);
			int pos_x = GetMin(GetMax(x + xp, 0), width - 1);
			mad += abs(tplate[y][x] - img[pos_y][pos_x]);
		}
	}
	return(mad / (dy * dx));
}

//MAD가 최소가 되는 위치 찾기
int Ex12_1()
{
	int height, width, dy, dx;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** tplate = (int**)ReadImage((char*)"lena_template.png", &dy, &dx);
	float min_mad = dy * dx * 255; // 나올 수 있는 최대값을 초기치로 설정
	int min_yp=0, min_xp=0;
	for (int yp = 0; yp < height; yp++) { // 모든 가능한 (xp, yp) 위치
		for (int xp = 0; xp < width; xp++) {
			float mad = MAD2(yp, xp, tplate, dy, dx, img, height, width);
			if (mad < min_mad) { // 구한 MAD가 현재까지의 MAD보다 작은 경우
				min_mad = mad; // MAD 업데이트 및 저장
				min_yp = yp; // 블록 위치 업데이트 및 저장
				min_xp = xp;
			}
		}
	}
	printf("\n (x, y) = (%d, %d), MAD = %.2f\n", min_xp, min_yp, min_mad);

	return 0;
}

//템플릿 매칭: 하나의 작은 이미지(템플릿)가 다른 큰 이미지의 어디에 위치하는지를 찾는 기술
void TemplateMatching(int** tplate, int dy, int dx, int** img, int height, int width,
	float* min_mad_out, int* min_yp_out, int* min_xp_out)
{
	float min_mad = INT_MAX;
	int min_yp = 0, min_xp = 0;
	for (int yp = 0; yp < height; yp++) {
		for (int xp = 0; xp < width; xp++) {
			float mad = MAD2(yp, xp, tplate, dy, dx, img, height, width);
			if (mad < min_mad) {
				min_mad = mad;
				min_yp = yp;
				min_xp = xp;
			}
		}
	}
	*min_mad_out = min_mad;
	*min_yp_out = min_yp;
	*min_xp_out = min_xp;
}

void DrawBox(int value, int y0, int x0, int dy, int dx, int** img, int height, int width)
{
	int y, x;
	for (x = 0; x < dx; x++) {
		// 상단 가로줄
		if (y0 < 0 || y0 >= height || x + x0 < 0 || x + x0 >= width) continue;
		else img[y0][x + x0] = value;
		// 하단 가로줄
		if (y0 + dy < 0 || y0 + dy >= height || x + x0 < 0 || x + x0 >= width) continue;
		else img[y0 + dy][x + x0] = value;
	}
	for (int y = 0; y < dy; y++) {
		// 좌측 세로줄
		if (y + y0 < 0 || y + y0 >= height || x0 < 0 || x0 >= width) continue;
		else img[y + y0][x0] = value;
		// 우측 세로줄
		if (y + y0 < 0 || y + y0 >= height || x0 + dx < 0 || x0 + dx >= width) continue;
		else img[y + y0][x0 + dx] = value;
	}
}

int Ex12_2()
{
	int height, width, dy, dx;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);	//비교할 이미지 사이즈
	int** tplate = (int**)ReadImage((char*)"lena_template.png", &dy, &dx);	//dy,dx는 템플릿의 사이즈
	float min_mad;
	int min_yp, min_xp;
	TemplateMatching(tplate, dy, dx, img, height, width, &min_mad, &min_yp, &min_xp);
	printf("\n (x, y) = (%d, %d), MAD = %.2f\n", min_xp, min_yp, min_mad);
	ImageShow((char*)"입력영상보기", img, height, width);
	DrawBox(255, min_yp, min_xp, dy, dx, img, height, width);
	ImageShow((char*)"출력영상보기", img, height, width);

	return 0;
}

//모자이크 -> db 영상 중 가장 유사도가 높은 db영상으로 교체

//db 영상 읽기#define DB_SIZE 510

//영상 일부 읽어 블록 만들기

#define DB_SIZE 510

//영상의 일부 읽어 블록 만들기
//y,x: 블록 시작 위치, dy,dx: 블록 사이즈
void ReadBlock(int** img, int y, int x, int dy, int dx, int** block)
{
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			block[i][j] = img[y + i][x + j];
		}
	}
}
void Ex12_3()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int dy = 128, dx = 128;
	int** block = (int**)IntAlloc2(dy, dx);
	int y = 256, x = 256;
	ReadBlock(img, y, x, dy, dx, block);
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"블록영상보기", block, dy, dx);
}

//입력 영상의 첫 번째 블록과 510개 db영상 중 가장 유사한 db 찾기
int FindBestDB(int** block, int** db[DB_SIZE], int dy, int dx)
{
	//첫 블록과 db영상 각각에 MAD를 계산하고 최소 MAD와 상응하는 db 번호 저장
	int min_index = 0;
	float min_mad = dy * dx * 255; // 큰 값으로 초기화
	for (int index = 0; index < DB_SIZE; index++) {
		float mad = MAD(block, db[index], dy, dx);
		if (mad < min_mad) {
			min_mad = mad;
			min_index = index;
		}
	}
	return(min_index);
}
//DB영상 읽어오기
void ReadAllDBImages(int** tplate[DB_SIZE])
{
	char filename[100];
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf(filename, ".\\dbs%04d.jpg", i);
		tplate[i] = (int**)ReadImage(filename, &dy, &dx);
	}
}

void WriteBlock(int** img, int y, int x, int dy, int dx, int** block)
{
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			img[y + i][x + j] = block[i][j];
		}
	}
}
//블록과 유사한 db 찾기
void Ex12_4()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** db[DB_SIZE];
	ReadAllDBImages(db); // 510장 db 영상 읽기
	int dy = 32, dx = 32;
	int** block = (int**)IntAlloc2(dy, dx);
	int y = 0, x = 0; // 첫번째 블록 시작 좌표
	ReadBlock(img, y, x, dy, dx, block); // 첫번째 블록 읽기
	int min_index = FindBestDB(block, db, dy, dx);
	printf("\n min_index = %d", min_index);
}

void Ex12_5()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** db[DB_SIZE];
	ReadAllDBImages(db);
	
	int dy = 32, dx = 32;
	int** block = (int**)IntAlloc2(dy, dx);

	for (int y = 0; y < height; y += dy) {
		for (int x = 0; x < width; x += dx) {
			ReadBlock(img, y, x, dy, dx, block);
			int min_index = FindBestDB(block, db, dy, dx);
			WriteBlock(img_out, y, x, dy, dx, db[min_index]);
		}
	}
	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}


// 학교 수업 시간 코드

void ReadBlock(int** img, int y0, int x0, int dy, int dx, int** block)
{
	for (int y = 0; y < dy; y++)
	{
		for (int x = 0; x < dx; x++)
		{
			block[y][x] = img[y0 + y][x0 + x];
		}
	}
}

void WriteBlock(int** img, int y0, int x0, int dy, int dx, int** block)
{
	for (int y = 0; y < dy; y++)
	{
		for (int x = 0; x < dx; x++)
		{
			img[y0 + y][x0 + x] = block[y][x];
		}
	}
}

int Ex1116_1()
{
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int dy = 32, dx = 32;
	int** block = (int**)IntAlloc2(dy, dx);

	ReadBlock(img, 100, 200, dy, dx, block);
	WriteImage((char*)"tmpelate.jpg", block, dy, dx);

	ImageShow((char*)"Input", img, height, width);
	ImageShow((char*)"Output", img_out, height, width);
	return 0;
}



void CopyImage(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = img[y][x];
		}
	}
}

void DrawBBox(int** img, int height, int width, int y0, int x0, int dy, int dx, int** img_out)
{
	CopyImage(img, height, width, img_out);

	for (int x = 0; x < dx; x++)
	{
		img_out[y0][x0 + x] = 255;
		img_out[y0 + dy][x0 + x] = 255;
	}
	for (int y = 0; y < dy; y++)
	{
		img_out[y0 + y][x0] = 255;
		img_out[y0 + y][x0 + dx] = 255;
	}
}

struct Point2D
{
	int y, x;
};


Point2D TemplateMatching(int** img, int height, int width, int dy, int dx, int** tmp)
{
	int** block = (int**)IntAlloc2(dy, dx);
	float mad_min = INT_MAX;
	int y_min = 0, x_min = 0;

	for (int y0 = 0; y0 < height - dy; y0++)
	{
		for (int x0 = 0; x0 < width - dx; x0++)
		{
			ReadBlock(img, y0, x0, dy, dx, block);

			float mad = MAD(block, tmp, dy, dx);

			if (mad < mad_min)
			{
				mad_min = mad;
				y_min = y0;
				x_min = x0;
			}
		}
	}
	IntFree2(block, dy, dx);

	Point2D P;
	P.x = x_min;
	P.y = y_min;

	return P;
}

int Ex1116_2()
{
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int dy, dx;
	int** tmp = ReadImage((char*)"template.jpg", &dy, &dx);
	int** img_out = (int**)IntAlloc2(height, width);

	Point2D P = TemplateMatching(img, height, width, dy, dx, tmp);
	DrawBBox(img, height, width, P.y, P.x, dy, dx, img_out);

	ImageShow((char*)"Input", img, height, width);
	ImageShow((char*)"template", tmp, dy, dx);
	ImageShow((char*)"Output", img_out, height, width);

	return 0;
}

#define M 510

//MAD가장 작은 db사진 찾기
int FindOptIndex(int** block, int*** db, int dy, int dx, int db_size)
{
	float mad_min = INT_MAX;
	int save_i = 0;
	for (int i = 0; i < db_size; i++)
	{
		float mad = MAD(block, db[i], dy, dx);
		if (mad < mad_min)
		{
			mad_min = mad;
			save_i = i;
		}
	}
	return save_i;
}

//모자이크 이미지 생성
void MosaicSingleBlock(int y0, int x0, int dy, int dx, int db_size, int** db[], int** img, int** img_out)	//int***db로 대체가능
{
	int** block = (int**)IntAlloc2(dy, dx);

	ReadBlock(img, y0, x0, dy, dy, block);

	int opt_index = FindOptIndex(block, db, dy, dx, db_size);

	WriteBlock(img_out, y0, x0, dy, dx, db[opt_index]);

	IntFree2(block, dy, dx);
}

//전체 영상에 대해 모자이크 실행
void MosaicImage(int dy, int dx, int db_size, int*** db, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y += dy)
	{
		for (int x = 0; x < width; x += dx)
		{
			MosaicSingleBlock(y, x, dy, dx, M, db, img, img_out);
		}
	}
}

//다운샘플링 결과: s_db에 저장
void MakeSDB(int*** db, int dy, int dx, int d, int*** s_db)
{
	for (int i = 0; i < M; i++)
	{
		if (i % 2 == 0)
		{
			for (int y = 0; y < dy; y += 2)
			{
				for (int x = 0; x < dx; x += 2)
				{
					s_db[i][y / 2][x / 2] = db[i][y][x];  // 값 복사
				}
			}
		}
	}
}

int Ex1124()
{
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int** db[M];
	char db_name[100];
	int dy, dx;

	for (int i = 0; i < M; i++)
	{
		sprintf(db_name, "./db영상(얼굴)/dbs%04d.jpg", i);
		db[i] = ReadImage(db_name, &dy, &dx);
		//ImageShow((char*)"db", db[i], dy, dx);
	}

	int** s_db[M];
	for (int i = 0; i < M; i++)
	{
		s_db[i] = (int**)IntAlloc2(dy / 2, dx / 2);
	}

	MakeSDB(db, dy, dx, M, s_db);

	//MosaicImage(dy, dx, M, db, img, height, width, img_out);
	MosaicImage(dy / 2, dx / 2, M, s_db, img, height, width, img_out);

	ImageShow((char*)"Output", img_out, height, width);

	return 0;
}

//칼라 이미지 모자이크
//int_rgb 구조체는 맨 위에 있음

//rgb값을 block에 저장
void ReadColorBlock(int_rgb** img, int y0, int x0, int dy, int dx, int_rgb** block)
{
	for (int y = 0; y < dy; y++)
	{
		for (int x = 0; x < dx; x++)
		{
			block[y][x] = img[y0 + y][x0 + x];	//c++에서만 가능, c에서는 block[y][x].r=img[y0+Y][x0+x].r;
		}
	}
}

//블록을 img에 저장
void WriteColorBlock(int_rgb** img, int y0, int x0, int dy, int dx, int_rgb** block)
{
	for (int y = 0; y < dy; y++)
	{
		for (int x = 0; x < dx; x++)
		{
			img[y0 + y][x0 + x] = block[y][x];
		}
	}
}

//두 개의 칼라 블록 간의 MAD 계산
float ColorMAD(int_rgb** A, int_rgb** B, int dy, int dx)
{
	float mad = 0.0;

	for (int y = 0; y < dy; y++)
	{
		for (int x = 0; x < dx; x++)
		{
			mad += abs(A[y][x].r - B[y][x].r) + abs(A[y][x].g - B[y][x].g) + abs(A[y][x].b - B[y][x].b);
		}
	}

	return mad / (dy * dx);
}

//칼라 블록과 db 영상 간의 MAD 비교하여 가장 작은 db 사진 인덱스 반환
int FindOptIndexColor(int_rgb** block, int_rgb*** db, int dy, int dx, int db_size)
{
	float mad_min = INT_MAX;
	int save_i = 0;
	for (int i = 0; i < db_size; i++)
	{
		float mad = ColorMAD(block, db[i], dy, dx);
		if (mad < mad_min)
		{
			mad_min = mad;
			save_i = i;
		}
	}
	return save_i;
}


void ColorMosaicSingleBlock(int y0, int x0, int dy, int dx, int db_size, int_rgb** db[], int_rgb** img, int_rgb** img_out)	//int***db로 대체가능
{
	//블록 생성
	int_rgb** block = (int_rgb**)IntColorAlloc2(dy, dx);
	
	//칼라 이미지 내에서 블록을 만들어냄
	ReadColorBlock(img, y0, x0, dy, dy, block);
	
	//블록과 db영상 간의 MAD를 비교하여 가장 작은 MAD를 가진 이미지의 인덱스를 선택
	int opt_index = FindOptIndexColor(block, db, dy, dx, db_size);

	//db영상을 img_out에 저장
	WriteColorBlock(img_out, y0, x0, dy, dx, db[opt_index]);

	//블록 메모리 해제
	IntColorFree2(block, dy, dx);
}

//전체 이미지에 모자이크 이미지 생성
void ColorMosaicImage(int dy, int dx, int db_size, int_rgb*** db, int_rgb** img, int height, int width, int_rgb** img_out)
{
	for (int y = 0; y < height; y += dy)
	{
		for (int x = 0; x < width; x += dx)
		{
			ColorMosaicSingleBlock(y, x, dy, dx, M, db, img, img_out);
		}
	}
}

//db이미지에서 짝수 번째 이미지를 선택하여 해당 이미지를 2배 다운샘플링하여 s_db에 저장
void MakeColorSDB(int_rgb*** db, int dy, int dx, int d, int_rgb*** s_db)		//s_db 리턴
{
	for (int i = 0; i < M; i++)
	{
		if (i % 2 == 0)
		{

			for (int y = 0; y < dy; y += 2)
			{
				for (int x = 0; x < dx; x += 2)
				{
					s_db[i][y / 2][x / 2] = db[i][y][x];  // 값 복사
				}
			}
		}
	}
}

//4배 다운샘플링
void MakeColorSDB4x4(int_rgb*** db, int dy, int dx, int d, int_rgb*** s_db)
{
	for (int i = 0; i < M; i++)
	{
		if (i % 2 == 0)
		{
			for (int y = 0; y < dy; y += 4)  // 이미지의 높이를 4로 나누어 4분의 1 크기로 다운샘플링
			{
				for (int x = 0; x < dx; x += 4)  // 이미지의 너비를 4로 나누어 4분의 1 크기로 다운샘플링
				{
					s_db[i][y / 4][x / 4] = db[i][y][x];  // 값 복사
				}
			}
		}
	}
}

int Ex12_9()
{
	int height, width;
	int_rgb** img = ReadColorImage((char*)"sample.png", &height, &width);
	int_rgb** img_out = (int_rgb**)IntColorAlloc2(height, width);

	int_rgb** db[M];
	char db_name[100];
	int dy, dx;

	//db영상 파일 읽어서 db[]에 저장
	for (int i = 0; i < M; i++)
	{
		sprintf(db_name, "./db영상(얼굴)/dbs%04d.jpg", i);
		db[i] = ReadColorImage(db_name, &dy, &dx);
		//ImageShow((char*)"db", db[i], dy, dx);
	}

	//s_db[]생성
	int_rgb** s_db[M];
	for (int i = 0; i < M; i++)
	{
		s_db[i] = (int_rgb**)IntColorAlloc2(dy / 2, dx / 2);
	}

	MakeColorSDB(db, dy, dx, M, s_db);

	//ColorMosaicImage(dy, dx, M, db, img, height, width, img_out);

	ColorMosaicImage(dy / 2, dx / 2, M, s_db, img, height, width, img_out);

	ColorImageShow((char*)"Input", img, height, width);
	ColorImageShow((char*)"Output", img_out, height, width);

	return 0;
}

//선 칼라로 그리기
#include <cmath>

typedef struct {
	int r, g, b;
} ColorRGB;

void DrawLineRGB(ColorRGB color, int x0, int y0, int x1, int y1, float thickness, ColorRGB*** img_out, int height, int width)
{
	float a = y1 - y0;
	float b = -(x1 - x0);
	float c = -(y1 - y0) * x0 + (x1 - x0) * y0;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (y < std::min(y1, y0) || y > std::max(y1, y0) || x < std::min(x1, x0) || x > std::max(x1, x0))
				continue;

			if (std::fabs(a * x + b * y + c) / std::sqrt(a * a + b * b) <= thickness) {
				img_out[y][x]->r = color.r;
				img_out[y][x]->g = color.g;
				img_out[y][x]->b = color.b;
			}
		}
	}
}

void myProb1()
{
	int height = 512, width = 512;

	// Allocate memory for RGB image
	ColorRGB*** img_out = new ColorRGB **[height];
	for (int i = 0; i < height; ++i) {
		img_out[i] = new ColorRGB * [width];
		for (int j = 0; j < width; ++j) {
			img_out[i][j] = new ColorRGB;
		}
	}

	int x0 = 100, y0 = 100, x1 = 400, y1 = 400;
	float thickness = 20;
	ColorRGB lineColor = { 255, 0, 0 }; // Red line color

	DrawLineRGB(lineColor, x0, y0, x1, y1, thickness, img_out, height, width);

	// Display or process the RGB image as needed
	// Example: ImageShowRGB("output", img_out, height, width);

	// Don't forget to free the allocated memory
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			delete img_out[i][j];
		}
		delete[] img_out[i];
	}
	delete[] img_out;
}


void DrawBBox(int_rgb** img, int height, int width, int_rgb** img_out)
{
	int x_max = -INT_MAX;
	int x_min = INT_MAX;
	int y_max = -INT_MAX;
	int y_min = INT_MAX;

	// 찾아진 영역의 경계 좌표 계산
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = img[y][x];

			// 픽셀 값이 128보다 큰 경우 경계 좌표 업데이트
			if (img[y][x].r > 128) {
				x_max = std::max(x_max, x);
				x_min = std::min(x_min, x);
				y_max = std::max(y_max, y);
				y_min = std::min(y_min, y);
			}
		}
	}

	float thickness = 2.0;

	// 빨간색 RGB 값 설정
	int_rgb red;
	red.r = 255;
	red.g = 0;
	red.b = 0;

	// 경계를 표시하는 선 그리기
	DrawLine(red, y_min, x_min, y_max, x_min, thickness, img_out, height, width);
	DrawLine(red, y_min, x_max, y_max, x_max, thickness, img_out, height, width);
	DrawLine(red, y_min, x_min, y_min, x_max, thickness, img_out, height, width);
	DrawLine(red, y_max, x_min, y_max, x_max, thickness, img_out, height, width);
}
int Ex12_2()
{
	int height, width, dy, dx;
	int_rgb** img = ReadColorImage("lena.png", &height, &width);   // Assuming ReadColorImage for RGB images
	int_rgb** tplate = ReadColorImage("lena_template.png", &dy, &dx);

	float min_mad;
	int min_yp, min_xp;
	TemplateMatching(tplate, dy, dx, img, height, width, &min_mad, &min_yp, &min_xp);

	printf("\n (x, y) = (%d, %d), MAD = %.2f\n", min_xp, min_yp, min_mad);

	ColorImageShow("입력영상보기", img, height, width);
	DrawBBox(img, height, width, min_yp, min_xp, dy, dx);  // Assuming DrawBBox for RGB images
	ColorImageShow("출력영상보기", img, height, width);

	// Free the allocated memory
	IntColorFree2(img, height, width);
	IntColorFree2(tplate, dy, dx);

	return 0;
}
