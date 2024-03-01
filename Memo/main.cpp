#include <windows.h>
#include <stdlib.h>
#include <imm.h>
#include "head.h" // 리소스 헤더 포함

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

HINSTANCE hInst;
HWND hEdit;
WCHAR *textBuffer = NULL;
int textLength = 0;
int caretPos = 0;
const int BUFFER_SIZE = 4096;
const int TEXT_HEIGHT = 16;

// 새 창을 생성하는 함수
HWND CreateNewMemoWindow(HINSTANCE hInstance) {
	textLength = 0; //기존 창을 복제해서 생성 되는 창을 초기에 텍스트를 없애고 시작함
    WNDCLASS wc = {0};

    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = TEXT("MyMemoClass");

    RegisterClass(&wc);

    HWND hWnd = CreateWindow(wc.lpszClassName, TEXT("동오의 비밀의 새 창 메모장"), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 640, 480, NULL, LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)), hInstance, NULL);
    
	ShowWindow(hWnd, SW_SHOWNORMAL); // "중요" 새 창을 화면에 표시함.

    // 텍스트 입력 받는거.
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return hWnd;
}

// WinMain 함수
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    MSG msg;
    WNDCLASS wc = {0};

    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = TEXT("MyMemoClass");

    if (!RegisterClass(&wc)) return 0;

    HWND hWnd = CreateWindow(
    wc.lpszClassName,                     // 윈도우 클래스 이름
    TEXT("동오의 비밀의 메모장"),   // 윈도우 제목
    WS_OVERLAPPEDWINDOW | WS_VSCROLL | WS_HSCROLL, // 윈도우 스타일에 수직 및 수평 스크롤바 추가 기능 추가해야지 구현됌
    CW_USEDEFAULT,                        // x 위치
    CW_USEDEFAULT,                        // y 위치
    640,                                  // 너비
    480,                                  // 높이
    NULL,                                 // 부모 윈도우 핸들
    LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)), // 메뉴 핸들
    hInstance,                            // 인스턴스 핸들
	NULL                                  // 추가 매개변수
	);

    if (!hWnd) return 0;

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    textBuffer = (WCHAR*)malloc((BUFFER_SIZE + 1) * sizeof(WCHAR));
    if (textBuffer == NULL) {
        MessageBox(NULL, TEXT("메모리 할당에 실패했습니다."), TEXT("에러"), MB_OK | MB_ICONERROR);
        return 0;
    }
    textBuffer[0] = L'\0';

    // 텍스트 입력 받는거.
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    free(textBuffer);
    return (int)msg.wParam;
}

// 윈도우 프로시저
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_CREATE:
            CreateCaret(hWnd, NULL, 1, TEXT_HEIGHT);
            ShowCaret(hWnd);
            SetCaretPos(0, 0);
            caretPos = 0;
            break;

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case ID_FILE_NEW:
                    textBuffer[0] = L'\0';
                    textLength = 0;
                    caretPos = 0;
					SetCaretPos(0, 0);
                    InvalidateRect(hWnd, NULL, TRUE);
                    break;
                
                case ID_FILE_NEW_WINDOW:
                    // 새 창 열기
                    CreateNewMemoWindow(hInst);
                    break;
				case ID_FILE_SAVE:
					{
						OPENFILENAME ofn;
						WCHAR szFileName[MAX_PATH] = L"";

						ZeroMemory(&ofn, sizeof(ofn));
						ofn.lStructSize = sizeof(ofn);
						ofn.hwndOwner = hWnd;
						ofn.lpstrFilter = L"텍스트 파일 (*.txt)\0*.TXT\0모든 파일 (*.*)\0*.*\0";
						ofn.lpstrFile = szFileName;
						ofn.lpstrFile[0] = L'\0';
						ofn.nMaxFile = sizeof(szFileName) / sizeof(*szFileName);
						ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_OVERWRITEPROMPT;

						if (GetSaveFileName(&ofn) == TRUE) {
							HANDLE hFile = CreateFile(ofn.lpstrFile, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
							if (hFile != INVALID_HANDLE_VALUE) {
								DWORD dwBytesWritten;

								WriteFile(hFile, textBuffer, textLength * sizeof(WCHAR), &dwBytesWritten, NULL);
								CloseHandle(hFile);
							}
						}
					}
					break;

					case ID_FILE_OPEN:
						{
							OPENFILENAME ofn;
							WCHAR szFileName[MAX_PATH] = L"";

							ZeroMemory(&ofn, sizeof(ofn));
							ofn.lStructSize = sizeof(ofn);
							ofn.hwndOwner = hWnd;
							ofn.lpstrFilter = L"텍스트 파일 (*.txt)\0*.TXT\0모든 파일 (*.*)\0*.*\0";
							ofn.lpstrFile = szFileName;
							ofn.lpstrFile[0] = L'\0';
							ofn.nMaxFile = sizeof(szFileName) / sizeof(*szFileName);
							ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

							if (GetOpenFileName(&ofn) == TRUE) {
								HANDLE hFile = CreateFile(ofn.lpstrFile, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
								if (hFile != INVALID_HANDLE_VALUE) {
									DWORD dwFileSize = GetFileSize(hFile, NULL);
									if (dwFileSize != INVALID_FILE_SIZE) {
										WCHAR* tempBuffer = (WCHAR*)malloc((dwFileSize + 1) * sizeof(WCHAR));
										if (tempBuffer != NULL) {
											DWORD dwBytesRead;
											ReadFile(hFile, tempBuffer, dwFileSize * sizeof(WCHAR), &dwBytesRead, NULL);
											tempBuffer[dwFileSize / sizeof(WCHAR)] = L'\0';
											// 이전 버퍼 해제

											free(textBuffer);
											// 새로운 버퍼 할당 및 복사
											textBuffer = (WCHAR*)malloc((dwFileSize + 1) * sizeof(WCHAR));
											if (textBuffer != NULL) {

												wcscpy_s(textBuffer, dwFileSize + 1, tempBuffer);

												textLength = (int)(dwFileSize / sizeof(WCHAR));
 
												caretPos = textLength; // 커서를 끝으로 이동
											}
											free(tempBuffer);
											InvalidateRect(hWnd, NULL, TRUE); // 윈도우를 다시 그리도록 강제
										}
									}
									CloseHandle(hFile);

								}

							}

						}

						break;

                case ID_FILE_EXIT:
                    DestroyWindow(hWnd);
                    break;

            }
            break;

        case WM_SIZE:
            break;

        case WM_KEYDOWN:
            switch (wParam) {
                case VK_HOME:
                    caretPos = 0;
                    break;
                
                case VK_END:
                    caretPos = textLength;
                    break;
                
                case VK_LEFT:
                    if (caretPos > 0) caretPos--;
                    break;
                
                case VK_RIGHT:
                    if (caretPos < textLength) caretPos++;
                    break;
				/*case VK_TAB:

					// 탭 문자를 삽입합니다.

					if (textLength < BUFFER_SIZE - 1) {

						for (int i = textLength; i > caretPos; i--) {

							textBuffer[i] = textBuffer[i - 1];
						}

						textBuffer[caretPos] = L'\t'; // 탭 문자 삽입

						caretPos++;

						textLength++;

					}

					break;
*/
                case VK_DELETE:
                    if (caretPos < textLength) {
                        for (int i = caretPos; i < textLength; i++) {
                            textBuffer[i] = textBuffer[i + 1];
                        }
                        textLength--;
                    }
                    break;

            }

            SetCaretPos(8 * caretPos, TEXT_HEIGHT * (caretPos / 80));
            InvalidateRect(hWnd, NULL, TRUE);
            break;

        case WM_CHAR:
            if (textLength < BUFFER_SIZE) {
                if (wParam == VK_BACK && caretPos > 0) {
                    caretPos--;
                    for (int i = caretPos; i < textLength; i++) {
                        textBuffer[i] = textBuffer[i + 1];
                    }
                    textLength--;
                } else if (wParam == VK_RETURN) {
                    if (textLength < BUFFER_SIZE - 1) {
                        for (int i = textLength; i > caretPos; i--) {
                            textBuffer[i] = textBuffer[i - 1];
                        }
                        textBuffer[caretPos] = L'\n';
                        caretPos++;
                        textLength++;
                    }
                } else {
                    if ((wParam >= 0x20 && wParam <= 0x7E) || ((wParam >= 0xAC00 && wParam <= 0xD7AF) && IsWindowEnabled(hWnd))) {
                        for (int i = textLength; i > caretPos; i--) {
                            textBuffer[i] = textBuffer[i - 1];
                        }
                        textBuffer[caretPos] = wParam;
                        caretPos++;
                        textLength++;
                    }
                }
            }

            SetCaretPos(8 * caretPos, TEXT_HEIGHT * (caretPos / 80));
            InvalidateRect(hWnd, NULL, TRUE);
            break;

        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            SetTextColor(hdc, RGB(0, 0, 0));
            SetBkColor(hdc, RGB(255, 255, 255));
            TextOut(hdc, 0, 0, textBuffer, textLength);
            EndPaint(hWnd, &ps);
            break;
        }

        case WM_DESTROY:
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}
