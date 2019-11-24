#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

#define garo 1280
#define sero 720
#define restrict_w ((garo) * (0.4))
#define restrict_h ((sero) * (0.4))
#define restrict_nan ((garo) * (0.9))
//#define K_size ((garo) * (0.022))

int kernel_size = 28;




int main()
{
	// 원본 사진 경로
	const char* file_path = "../card_img/*.jpg";


	vector<String> filename;
	glob(file_path, filename, true);

	vector<Mat> roi_vector;


	for (int i = 0; i < filename.size(); i++)
	{
		String fname = filename[i];


		Mat frame = imread(fname);
		Mat last_result, last_result2, copy, copy_frame, matInput, matResult, k_3, k_15;

		//resize(frame, frame, Size(garo, sero));

		//imshow("현재 원본영상", frame);
		//waitKey(0);
		//destroyAllWindows();
		last_result = frame.clone();
		last_result2 = frame.clone();
		matInput = frame.clone();
		copy = frame.clone();
		copy_frame = frame.clone();
		k_3 = frame.clone();
		k_15 = frame.clone();

		Mat roi;

		// resize 후 원본영상을 복사
		cvtColor(matInput, matInput, COLOR_RGB2GRAY);
		bilateralFilter(matInput, matResult, 5, 75, 75);
		//GaussianBlur(matInput, matResult, Size(11,11), 0);
		Canny(matResult, matInput, 0, 50);
		//imshow("canny", matInput);
		//waitKey(0);
		//destroyAllWindows();


		//Mat Kernel_15 = getStructuringElement(MORPH_RECT, Size((int)K_size, 1));
		//Mat Kernel_15 = getStructuringElement(MORPH_RECT, Size(10, 1));
		Mat Erode_Kernel = getStructuringElement(MORPH_RECT, Size(1, 3));
		//Mat Dilate_Kernel = getStructuringElement(MORPH_RECT, Size(25, 5));
		Mat Dilate_Kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, 5));

		// 모폴로지에 사용할 커널 생성
		Mat dst1, dst2, dst3;
		//morphologyEx(matInput, dst1, MORPH_DILATE, Kernel_15, Point(-1, -1), 2);
		morphologyEx(matInput, dst2, MORPH_ERODE, Erode_Kernel, Point(-1, -1), 2);
		morphologyEx(dst2, dst3, MORPH_DILATE, Dilate_Kernel, Point(-1, -1), 2);

		//imshow("canny -> 팽창연산", dst1);
		//waitKey(0);
		//destroyAllWindows();

		//mshow("canny -> 침식연산", dst2);
		//aitKey(0);
		//estroyAllWindows();
		//
		//mshow("canny -> 침식연산 후 팽창", dst3);
		//aitKey(0);
		//estroyAllWindows();


		vector< vector<Point> > contour, contour1;

		findContours(dst3, contour, RETR_LIST, CHAIN_APPROX_NONE);

		vector<Rect> boundRect(contour.size());

		vector<RotatedRect> minRect(contour.size());

		// minAreaRect 생성
		for (int i = 0; i < contour.size(); i++)
		{
			minRect[i] = minAreaRect(Mat(contour[i]));
		}

		// 후보 3개 저장용 구조체
		typedef struct set
		{
			float area = 0;
			int idx = 0;
		}Set;

		vector <Set> temp_set(3); // 후보 3개

		// minAreaRect 그리는 부분
		for (int i = 0; i < contour.size(); i++)
		{
			// center에 사각형의 중심(x,y)좌표,  size에 width, height, 기울어진 각도를 가지고있음
			Point2f rect_points[4];
			minRect[i].points(rect_points);

			float w, h;
			w = minRect[i].size.width;
			h = minRect[i].size.height;


			float ratio = w > h ? w / h : h / w;

			ratio = abs(ratio);

			//전체 다 boxing
			for (int j = 0; j < 4; j++)
			{
				line(last_result2, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 255, 0), 4, 8);
			}


			//가로, 세로 길이 , 종횡비 등으로 거르기
			if (((w <= restrict_w && ratio >= 4) || (h <= restrict_h && ratio >= 4)) || ((isnan(ratio) && w < restrict_nan) || (isnan(ratio) && h < restrict_nan)))
			{
				int k = 0;
				//넓이 순으로 상위 3등까지만 저장
				if (temp_set[2].area < minRect[i].size.area())
				{
					if (temp_set[1].area < minRect[i].size.area())
					{
						if (temp_set[0].area < minRect[i].size.area())
						{
							temp_set[2] = temp_set[1];
							temp_set[1] = temp_set[0];

							temp_set[0].area = minRect[i].size.area();
							temp_set[0].idx = i;
						}

						else
						{
							temp_set[2] = temp_set[1];

							temp_set[1].area = minRect[i].size.area();
							temp_set[1].idx = i;
						}
					}
					else
					{
						temp_set[2].area = minRect[i].size.area();
						temp_set[2].idx = i;
					}

				}
				/*조건 박스
				for (int j = 0; j < 4; j++)
				{
				   line(k_15, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 4, 8);
				}
				*/
			}


			//전체 중에서 맨 마지막 컨투어 일때 
			if (i == contour.size() - 1)
			{
				for (int k = 0; k < 2; k++) // 2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222를 3으로 바꿔야 후보 3개 지금은 2로함
				{

					//cout << "인덱스 : " << temp_set[k].idx;
					//cout << " 의 넓이 : " << temp_set[k].area << endl;




					minRect[temp_set[k].idx].points(rect_points);

					int left_idx1 = 0;
					int left_idx2 = 0;
					int right_idx1 = 0;
					int right_idx2 = 0;

					for (int j = 0; j < 4; j++)
					{

						line(k_15, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 4, 8);
						//cout << "현재의 j : " << j << " rect_points[j] : " << rect_points[j] << endl;

						if (rect_points[left_idx2].x >= rect_points[j].x)
						{
							if (rect_points[left_idx1].x >= rect_points[j].x)
							{
								left_idx2 = left_idx1;
								left_idx1 = j;
								continue;
							}

							left_idx2 = j;

						}

						if (rect_points[right_idx2].x <= rect_points[j].x)
						{
							if (rect_points[right_idx1].x <= rect_points[j].x)
							{
								right_idx2 = right_idx1;
								right_idx1 = j;
								continue;
							}

							right_idx2 = j;
						}

					}


					int leftTop = rect_points[left_idx2].y > rect_points[left_idx1].y ? left_idx1 : left_idx2;
					int leftBottom = left_idx1 == leftTop ? left_idx2 : left_idx1;

					int rightTop = rect_points[right_idx2].y > rect_points[right_idx1].y ? right_idx1 : right_idx2;
					int rightBottom = right_idx1 == rightTop ? right_idx2 : right_idx1;

					float width, height;

					width = sqrtf(powf(rect_points[rightTop].x - rect_points[leftTop].x, 2) + powf(abs(rect_points[rightTop].y - rect_points[leftTop].y), 2));
					height = sqrtf(powf(abs(rect_points[leftTop].x - rect_points[leftBottom].x), 2) + powf(rect_points[leftBottom].y - rect_points[leftTop].y, 2));

					//cout << "left top : " << leftTop << endl;
					//cout << "left bottom : " << leftBottom << endl;
					//cout << "right top : " << rightTop << endl;
					//cout << "right bottom : " << rightBottom << endl;

					Point2f src[4], dst[4];

					src[0] = Point2f(rect_points[leftTop].x, rect_points[leftTop].y - 10);
					src[1] = Point2f(rect_points[rightTop].x, rect_points[rightTop].y - 10);
					src[2] = Point2f(rect_points[leftBottom].x, rect_points[leftBottom].y + 10);
					src[3] = Point2f(rect_points[rightBottom].x, rect_points[rightBottom].y + 10);

					dst[0] = Point2f(0, 0);
					dst[1] = Point2f(width - 1, 0);
					dst[2] = Point2f(0, height - 1);
					dst[3] = Point2f(width - 1, height - 1);

					Mat transformMatrix = getPerspectiveTransform(src, dst);

					warpPerspective(last_result, roi, transformMatrix, Size(width, height));

					if (roi.empty())
					{
						cout << "roi is empty!" << endl;
						continue;
					}

					imwrite("./roi.jpg", roi);
					Mat read_roi = imread("./roi.jpg");
					Mat filter, canny, thre, rest, med;

					// 화질 개선된 grayscale 이미지가 filter에 저장됨

					cvtColor(read_roi, read_roi, COLOR_RGB2GRAY);
					//imshow("저장된 roi", read_roi);
					//waitKey();
					GaussianBlur(read_roi, filter, Size(0, 0), 3);
					addWeighted(read_roi, 5.0, filter, -4.0, 0, filter);
					//imshow("filter", filter);
					//waitKey();
					//destroyAllWindows();

					roi_vector.push_back(filter.clone());


				}
			}
		}

		// roi 이미지 저장
		for (int i = 0; i < roi_vector.size(); i++)
		{
			char roi_name[100];
			sprintf_s(roi_name, "../roi_img/%d.jpg", i);
			Mat roi = roi_vector[i];
			imwrite(roi_name, roi);
		}

		cout << i << "번째 이미지 저장 끝" << endl;
	}
}
