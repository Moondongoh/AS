/*
[END]1. 윈도우 창을 생성한다.
[END]>> 상단 툴바에 파일 서식 작업 크기가 있으며 각 기능들 추가 중이다.
[---]1-1. 새로 만들기, 새 창, 저장 및 열기, 끝내기 기능 추가 할 예정
[END]++ 툴바에 도형 칸 넣고 도형 그리기 추가 할 예정
[END]2. 마우스 좌 클릭 시 그리기 가능하다.
[END]3. 마우스 우 클릭 시 지우기 가능하다.
[END]4. 작업 영역(1920*1080)을 가지면 백 버퍼릉 만들어 더블 버퍼링 사용 중이다.
[NO]5. 스크롤 기능? 
[NO]6. 작업 영역에 격자 넣기?
*/
#include <windows.h>
#include <windef.h>
#include "resource.h"

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

void CreateNewPaintWindow(HINSTANCE hInstance);

LPCTSTR lpszClass = TEXT("PAINT");

COLORREF LineColor = 0x00000000;
COLORREF FullColor = 0x00000000;

POINT startPoint = {0, 0};
POINT endPoint = {0, 0};
POINT prevEndPoint = {0, 0};	// 이전의 끝점을 저장하는 변수 추가

BOOL isDrawing = FALSE;			// 기본 그리기
BOOL isDrawing_2 = FALSE;		// 메뉴바 그리기
BOOL isEraser = FALSE;			// 지우기
BOOL isRect = FALSE;			// 사각형
BOOL isEllipse = FALSE;			// 타원
BOOL isRed = FALSE;				// 빨간색
BOOL isBlue = FALSE;			// 파란색
BOOL isYellow = FALSE;			// 노란색
BOOL isGreen = FALSE;			// 초록색
BOOL isBlack= FALSE;			// 검정색
BOOL isZero= FALSE;				// 투명색

int cWidth;						//지우개 크기 변수

// 백 버퍼 관련 변수
HDC hdcBuffer = NULL;
HBITMAP hBitmapBuffer = NULL;
HBITMAP hBitmapOld = NULL;
HBRUSH hWhiteBrush;				// 백 버퍼를 하얀색으로 채우기 위한 브러시
RECT rect;						// 백 버퍼 크기를 정의하는 사각형

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{

    HWND hwnd;
    MSG msg;
    WNDCLASS WndClass;
    WndClass.style = CS_HREDRAW | CS_VREDRAW;
    WndClass.lpfnWndProc = WndProc;
    WndClass.cbClsExtra = 0;
    WndClass.cbWndExtra = 0;
    WndClass.hInstance = hInstance;
    WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    WndClass.lpszMenuName = NULL;
    WndClass.lpszClassName = lpszClass;
    RegisterClass(&WndClass);

    hwnd = CreateWindow(lpszClass,
                        lpszClass,
                        WS_OVERLAPPEDWINDOW,
                        100,
                        50,
                        600,
                        400,
                        NULL,
                        LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)),
                        hInstance,
                        NULL);
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
        
    // 메시지 루프가 종료되면, 백 버퍼 관련 리소스 해제
    if (hBitmapOld != NULL)
        SelectObject(hdcBuffer, hBitmapOld);

    if (hBitmapBuffer != NULL)
        DeleteObject(hBitmapBuffer);

    if (hdcBuffer != NULL)
        DeleteDC(hdcBuffer);
    //

    return (int)msg.wParam;
}

HDC hdc;

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    static HBITMAP hTempBitmap = NULL;								// 임시 백 버퍼를 저장하기 위한 변수
    static HDC hdcTemp = NULL;										// 임시 백 버퍼의 DC를 저장하기 위한 변수

    PAINTSTRUCT ps;

    switch (iMsg)
    {
    case WM_CREATE:
        // 백 버퍼 초기화
        hdcBuffer = CreateCompatibleDC(NULL);
        hBitmapBuffer = CreateCompatibleBitmap(GetDC(hwnd), 1920, 1080);
        hBitmapOld = (HBITMAP)SelectObject(hdcBuffer, hBitmapBuffer);
        hWhiteBrush = CreateSolidBrush(RGB(255, 255, 255));         // 백 버퍼 전체를 하얀색으로 채우기

        rect.left = 0;
        rect.top = 0;
        rect.right = 1920;
        rect.bottom = 1080;

        FillRect(hdcBuffer, &rect, hWhiteBrush);

        isDrawing = FALSE;
        isDrawing_2 = FALSE; 
        isEraser = FALSE;
        isRect = FALSE;
        isEllipse = FALSE;

        break;

    case WM_MOUSEMOVE:
        if (wParam == MK_LBUTTON ||wParam == MK_RBUTTON ){ // 이건 클릭 중인지 확인 -> 클릭중이면서 마우스 움직이면?
        if (isDrawing || isEraser || isRect || isEllipse)
        {
            prevEndPoint = endPoint;						// 이전 끝점을 현재 끝점으로 갱신
            endPoint.x = LOWORD(lParam);
            endPoint.y = HIWORD(lParam);
            InvalidateRect(hwnd, NULL, FALSE);
        }
        }
        break;

        //LEFT : 그리기 RIGHT : 지우기
    case WM_LBUTTONDOWN:
		if(isRect == TRUE || isEllipse == TRUE)
		{

        // 도형을 그릴 때마다 새로운 임시 백 버퍼를 생성
        if (hTempBitmap != NULL)
        {
            SelectObject(hdcTemp, hBitmapOld);
            DeleteObject(hTempBitmap);
        }

        hdcTemp = CreateCompatibleDC(hdcBuffer);							// 임시 DC생성
        hTempBitmap = CreateCompatibleBitmap(hdcBuffer, 1920, 1080);		// 화면 크기랑 동일한 사이즈
        hBitmapOld = (HBITMAP)SelectObject(hdcTemp, hTempBitmap);			// 임시 DC로 선택

        // 현재 화면을 임시 백 버퍼에 복사
        BitBlt(hdcTemp, 0, 0, 1920, 1080, hdcBuffer, 0, 0, SRCCOPY);
		}

        // 현재 마우스 위치를 시작점으로 설정
        startPoint.x = endPoint.x = LOWORD(lParam);
        startPoint.y = endPoint.y = HIWORD(lParam);
        prevEndPoint = startPoint;											// 이전 끝점을 현재 시작점으로 설정

        // 선 그리기 시작
        isDrawing = TRUE;
        break;

    case WM_LBUTTONUP:
        // 도형 그리기 종료
        isDrawing = FALSE;

        // 임시 백 버퍼의 내용을 화면에 출력
        BitBlt(hdcBuffer, 0, 0, 1920, 1080, hdcTemp, 0, 0, SRCCOPY);

        // 임시 백 버퍼 해제
        SelectObject(hdcTemp, hBitmapOld);
        DeleteObject(hTempBitmap);
        hTempBitmap = NULL;

        // 화면 갱신
        InvalidateRect(hwnd, NULL, FALSE);
        break;

    case WM_RBUTTONDOWN:
        // 지우개 모드로 설정
        isEraser = TRUE;
		isRect = FALSE;
		isEllipse = FALSE;
		isDrawing = TRUE;
        startPoint.x = endPoint.x = LOWORD(lParam);
        startPoint.y = endPoint.y = HIWORD(lParam);
        prevEndPoint = startPoint; // 이전 끝점을 현재 시작점으로 설정
        break;

    case WM_RBUTTONUP:
        // 지우개 모드 해제
        isEraser = FALSE;
        break;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        // 선 색상
        case ID_RED:
            LineColor = RGB(255, 0, 0);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_BLUE:
            LineColor = RGB(0, 0, 255);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_YELLOW:
            LineColor = RGB(255, 255, 0);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_GREEN:
            LineColor = RGB(0, 255, 0);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_BLACK:
            LineColor = RGB(0, 0, 0);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_Full_Red:
            FullColor = RGB(255, 0, 0);
            isRed = TRUE;
            break;

        case ID_Full_Blue:
            FullColor = RGB(0, 0, 255);
            isBlue = TRUE;
            break;

        case ID_Full_Yellow:
            FullColor = RGB(255, 255, 0);
            isYellow = TRUE;
            break;

        case ID_Full_Green:
            FullColor = RGB(0, 255, 0);
            isGreen = TRUE;
            break;

        case ID_Full_Black:
            FullColor = RGB(0, 0, 0);
            isBlack = TRUE;
            break;

        case ID_Full_Zero:
            isRed = FALSE;
            isBlue =FALSE;
            isYellow =FALSE;
            isGreen =FALSE;
            isBlack =FALSE;
			isZero = TRUE;
            break;

        // 작업 사이즈  
        case ID_SIZE_10:
            cWidth = 10;
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_SIZE_20:
            cWidth = 20;
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_SIZE_30:
            cWidth = 30;
            InvalidateRect(hwnd, NULL, FALSE);
            break;

		// 새로 만들기
        case ID_FILE_PAINT_NEW:
            FillRect(hdcBuffer, &rect, hWhiteBrush);
            InvalidateRect(hwnd, NULL, TRUE);
            isDrawing = FALSE;
            isDrawing_2 = FALSE;
            isRect = FALSE;
            isEllipse = FALSE;
            break;
		
		// 새 창 열기
        case ID_FILE_NEW_PAINT_WINDOW:
            CreateNewPaintWindow(GetModuleHandle(NULL));
            break;
		
		// 열기
        case ID_FILE_PAINT_OPEN:
            break;

		// 저장
        case ID_FILE_PAINT_SAVE:
            break;

        case ID_FILE_EXIT:
            DestroyWindow(hwnd);
            break;

		// 메뉴바 사각형 그리기
        case ID_Rectangle:
            isRect = TRUE;
            isDrawing = FALSE;
            isDrawing_2= FALSE;
            isEraser = FALSE;
            isEllipse = FALSE;
            break;

		// 메뉴바 선 그리기
        case ID_Line:
            isDrawing_2 = TRUE;
            isRect = FALSE;
            isEllipse = FALSE;
            break;

		// 메뉴바 원 그리기
        case ID_Ellipse:
            isEllipse = TRUE;
            isDrawing = FALSE;
			isDrawing_2= FALSE;
            isEraser = FALSE;
            isRect =FALSE;
            break;

        }
        break;

    case WM_PAINT:
        {
            HDC hdc = BeginPaint(hwnd, &ps);

            // 백 버퍼에 그림을 그림
            HBRUSH hBrush;
            if (isEraser)
                hBrush = CreateSolidBrush(RGB(255, 255, 255)); // 지우개는 흰색으로 설정

            else if(isRed || isBlue || isYellow || isGreen || isBlack)
            {       
                hBrush = CreateSolidBrush(FullColor);
                SelectObject(hdcBuffer, hBrush);
            }

            else 
            {
                hBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
                SelectObject(hdcBuffer, hBrush);
            }

            if (isEllipse)
            {
                isDrawing = FALSE;
				//FillRect(hdcBuffer, &rect, hWhiteBrush);
                Ellipse(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);
                isEllipse = TRUE;
            }

            if (isRect)
            {
				//FillRect(hdcBuffer, &rect, hWhiteBrush);
                Rectangle(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);
                isRect = TRUE;
            }

            else if (isDrawing || isEraser || isDrawing_2)
            {
                if (isEraser)
                    SelectObject(hdcBuffer, CreatePen(PS_SOLID, cWidth, RGB(255, 255, 255))); // 펜 없이 그리기

                else if(isDrawing_2)
                    isDrawing = TRUE;

                else
                    SelectObject(hdcBuffer, CreatePen(PS_SOLID, cWidth, LineColor)); // 펜 선택
                MoveToEx(hdcBuffer, prevEndPoint.x, prevEndPoint.y, NULL);
                LineTo(hdcBuffer, endPoint.x, endPoint.y);
            }

            DeleteObject(hBrush);

            // 백 버퍼에 그린 그림을 화면에 복사
            BitBlt(hdc, 0, 0, 1920, 1080, hdcBuffer, 0, 0, SRCCOPY);

            EndPaint(hwnd, &ps);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    }

    return DefWindowProc(hwnd, iMsg, wParam, lParam);
}

void CreateNewPaintWindow(HINSTANCE hInstance)
{
    HWND newHwnd;
    MSG newMsg;
    WNDCLASS newWndClass;

    // 새로운 윈도우 클래스를 등록합니다.
    newWndClass.style = CS_HREDRAW | CS_VREDRAW;
    newWndClass.lpfnWndProc = WndProc;
    newWndClass.cbClsExtra = 0;
    newWndClass.cbWndExtra = 0;
    newWndClass.hInstance = hInstance;
    newWndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    newWndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    newWndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    newWndClass.lpszMenuName = NULL;
    newWndClass.lpszClassName = TEXT("NEW_PAINT_WINDOW");
    RegisterClass(&newWndClass);

    // 새로운 윈도우를 생성합니다.
    newHwnd = CreateWindow(TEXT("NEW_PAINT_WINDOW"),
                            TEXT("New Paint Window"),
                            WS_OVERLAPPEDWINDOW,
                            CW_USEDEFAULT,
                            CW_USEDEFAULT,
                            600,
                            400,
                            NULL,
                            LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)),
                            hInstance,
                            NULL);
    ShowWindow(newHwnd, SW_SHOW);
    UpdateWindow(newHwnd);

    // 새로운 윈도우의 메시지 루프를 실행합니다.
    while (GetMessage(&newMsg, NULL, 0, 0))
    {
        TranslateMessage(&newMsg);
        DispatchMessage(&newMsg);
    }
}