/*
* 1. 윈도우 창 생성이 가능하다.              [완료]
* - 윈도우 창 타이틀 이름 변경               [완료]
* - 윈도우 창 메뉴바 수정 및 추가            [완료]
* 2. 한글 및 영어 입력이 가능하다.           [완료]
* 3. 메뉴바 기능을 이용 가능하다.            [완료]
* - 새로 만들기, 새 창, 저장 및 열기, 끝내기 [완료]
* 4. 스크롤 기능을 이용 가능하다.            [완료]
* 5. 캐럿의 기능을 구현함.
* -EX)하단 혹은 우측 끝에 닿으면 화면 처리   [미완료]
*/

#include "pch.h"
#include "framework.h"
#include "MFC_Memo.h"
#include "MFC_MemoDoc.h"
#include "MFC_MemoView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMFCMemoView
IMPLEMENT_DYNCREATE(CMFCMemoView, CScrollView)
BEGIN_MESSAGE_MAP(CMFCMemoView, CScrollView)

    // 키보드 이벤트 핸들러 추가
    ON_WM_CHAR()
    // *********************************** 메뉴 ***********************************
    ON_COMMAND(ID_FILE_SAVE_2, &CMFCMemoView::OnFileSave2)
    ON_COMMAND(ID_FILE_OPEN, &CMFCMemoView::OnFileOpen)
    ON_COMMAND(ID_FILE_NEW, &CMFCMemoView::OnFileNew)
    ON_COMMAND(ID_FILE_NEW_WINDOW, &CMFCMemoView::OnFileNewWindow)
    // *********************************** 메뉴 ***********************************
    // 
    // *********************************** 캐럿 ***********************************
    ON_WM_SETFOCUS()
    ON_WM_KILLFOCUS()
    // *********************************** 캐럿 ***********************************
    // 
    // *********************************** 스크롤 ***********************************
    ON_WM_KEYDOWN()
    ON_WM_VSCROLL()
    ON_WM_HSCROLL()
    // *********************************** 스크롤 ***********************************
END_MESSAGE_MAP()

// CMFCMemoView 생성/소멸

CMFCMemoView::CMFCMemoView() noexcept :

    m_nCurrentRow(0),               // x초기화
    m_nCurrentColumn(0),            // y초기화
    m_bCaretVisible(false),         // 캐럿

    // *********************************** 스크롤 ***********************************
    m_nScrollWidth(5000),           // 임의의 너비 설정
    m_nScrollHeight(5000),          // 임의의 높이 설정
    m_nPageSize(100)                // 한 페이지의 크기 설정
    // *********************************** 스크롤 ***********************************
{
    for (int i = 0; i < Row; ++i) 
    {
        for (int j = 0; j < Columns; ++j) 
        {
            m_textArray[i][j] = '\0';
        }
    }
}

CMFCMemoView::~CMFCMemoView()
{
    // *********************************** 캐럿 ***********************************
    if (m_bCaretVisible)
    {
        HideCaret();
        DestroyCaret();
    }
    // *********************************** 캐럿 ***********************************
}

BOOL CMFCMemoView::PreCreateWindow(CREATESTRUCT& cs)
{
    return CScrollView::PreCreateWindow(cs);
}

// *********************************** 텍스트 입력 기능 ***********************************
void CMFCMemoView::OnDraw(CDC* pDC)
{
    CMFCMemoDoc* pDoc = GetDocument();
    ASSERT_VALID(pDoc);
    if (!pDoc)
        return;

    // 폰트 설정 (예시로 Arial, 크기 12 사용)
    CFont font;
    font.CreateFontW(20, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Arial");
    pDC->SelectObject(&font);

    // 텍스트를 화면에 출력합니다.
    for (int i = 0; i < Row; ++i)
    {
        CString textString(m_textArray[i]); // wchar_t 배열을 CString으로 변환
        pDC->TextOut(0, i * 20, textString); // CString을 사용하여 출력
    }
}

void CMFCMemoView::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
    if (nChar == VK_BACK)
    {
        if (m_nCurrentColumn > 0)
        {
            m_nCurrentColumn--;                                                         // 현재 커서 위치가 첫 번째 열이 아니면 현재 줄에서 이전 열로 이동합니다.
        }
        else if (m_nCurrentRow > 0)
        {
            m_nCurrentRow--;                                                            // 현재 커서 위치가 첫 번째 열이고, 현재 줄이 첫 번째 줄이 아니라면 이전 줄로 이동합니다.
            m_nCurrentColumn = wcslen(m_textArray[m_nCurrentRow]);                      // 이전 줄의 마지막 열로 이동합니다.

            int currentRowLength = wcslen(m_textArray[m_nCurrentRow]);                  // 현재 줄의 문자열 길이를 계산합니다.

            if (currentRowLength == 0)                                                  // 이전 줄이 있고, 현재 줄이 비어있으면 이전 줄을 지웁니다.
            {
                wcscpy_s(m_textArray[m_nCurrentRow], m_textArray[m_nCurrentRow + 1]);   // 현재 줄을 이전 줄로 복사합니다.

                wmemset(m_textArray[m_nCurrentRow + 1], L'\0', Columns);                // 이전 줄을 비웁니다.

                m_nCurrentColumn = currentRowLength;                                    // 커서를 이전 줄의 끝으로 이동합니다.
            }
        }

        if (m_textArray[m_nCurrentRow][m_nCurrentColumn] != L'\0')                      // 현재 커서 위치가 문자열의 끝이 아니라면 백스페이스 처리
        {
            m_textArray[m_nCurrentRow][m_nCurrentColumn] = L'\0';
        }
    }
    else if (nChar == VK_RETURN)
    {
        m_nCurrentRow++;
        m_nCurrentColumn = 0;
    }
    else if (nChar == VK_TAB)
    {
        m_textArray[m_nCurrentRow][m_nCurrentColumn] = L' ';
        m_nCurrentColumn++;
    }
    else if (nChar == VK_ESCAPE)
    {
    }
    else {
        m_textArray[m_nCurrentRow][m_nCurrentColumn] = (wchar_t)nChar;
        m_nCurrentColumn++;
    }
    UpdateCaretPosition();
    Invalidate();

    CView::OnChar(nChar, nRepCnt, nFlags);
}

void CMFCMemoView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
    // 방향키에 따라 캐럿의 위치를 업데이트
    switch (nChar)
    {
    case VK_UP:
        if (m_nCurrentRow > 0)
        {
            m_nCurrentRow--;
            m_nCurrentColumn = min(m_nCurrentColumn, wcslen(m_textArray[m_nCurrentRow]));
        }
        break;
    case VK_DOWN:
        if (m_nCurrentRow < Row - 1)
        {
            m_nCurrentRow++;
            m_nCurrentColumn = min(m_nCurrentColumn, wcslen(m_textArray[m_nCurrentRow]));
        }
        break;
    case VK_LEFT:
        if (m_nCurrentColumn > 0)
        {
            m_nCurrentColumn--;
        }
        else if (m_nCurrentRow > 0)
        {
            m_nCurrentRow--;
            m_nCurrentColumn = wcslen(m_textArray[m_nCurrentRow]);
        }
        break;
    case VK_RIGHT:
        if (m_nCurrentColumn < wcslen(m_textArray[m_nCurrentRow]))
        {
            m_nCurrentColumn++;
        }
        else if (m_nCurrentRow < Row - 1)
        {
            m_nCurrentRow++;
            m_nCurrentColumn = 0;
        }
        break;
    }
    UpdateCaretPosition();

    CScrollView::OnKeyDown(nChar, nRepCnt, nFlags);
}
// *********************************** 텍스트 입력 기능 ***********************************

#ifdef _DEBUG
void CMFCMemoView::AssertValid() const
{
    CView::AssertValid();
}

void CMFCMemoView::Dump(CDumpContext& dc) const
{
    CView::Dump(dc);
}

CMFCMemoDoc* CMFCMemoView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
    ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCMemoDoc)));
    return (CMFCMemoDoc*)m_pDocument;
}
#endif //_DEBUG

// *********************************** 메뉴바 기능 ***********************************

void CMFCMemoView::OnFileSave2()
{
    // 파일 저장 대화상자 생성
    CFileDialog fileDlg(FALSE, _T("txt"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("Text Files (*.txt)|*.txt||"), this);

    // 파일 저장 대화상자 열기
    if (fileDlg.DoModal() == IDOK)
    {
        CString pathName = fileDlg.GetPathName();

        // 파일 열기
        CFile file(pathName, CFile::modeCreate | CFile::modeWrite);
        CArchive ar(&file, CArchive::store);

        // 텍스트 데이터를 파일에 저장
        for (int i = 0; i < Row; ++i)
        {
            ar.WriteString(CString(m_textArray[i]) + _T("\n"));
        }

        ar.Close();
        file.Close();
    }
}

void CMFCMemoView::OnFileOpen()
{
    // 파일 열기 대화상자 생성
    CFileDialog fileDlg(TRUE, _T("txt"), NULL, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY, _T("Text Files (*.txt)|*.txt||"), this);

    // 파일 열기 대화상자 열기
    if (fileDlg.DoModal() == IDOK)
    {
        CString pathName = fileDlg.GetPathName();

        // 파일 열기
        CFile file(pathName, CFile::modeRead);
        CArchive ar(&file, CArchive::load);

        // 텍스트 데이터 초기화
        for (int i = 0; i < Row; ++i)
        {
            memset(m_textArray[i], '\0', Columns);
        }

        // 파일에서 텍스트 데이터 읽기
        int row = 0;
        CString line;
        while (ar.ReadString(line))
        {
            if (row < Row)
            {
                wcscpy_s(m_textArray[row], Columns, line.GetBuffer());
                line.ReleaseBuffer();
                row++;
            }
        }

        ar.Close();
        file.Close();

        // 텍스트 데이터 출력
        Invalidate();
    }
}

void CMFCMemoView::OnFileNew()
{
    // 텍스트 데이터 초기화
    for (int i = 0; i < Row; ++i)
    {
        memset(m_textArray[i], '\0', Columns);
    }

    // 커서 위치 초기화
    m_nCurrentRow = 0;
    m_nCurrentColumn = 0;

    // 텍스트 데이터 출력
    Invalidate();
}

void CMFCMemoView::OnFileNewWindow()
{
    // 애플리케이션 인스턴스를 가져옵니다.
    CWinApp* pApp = AfxGetApp();

    // 첫 번째 문서 템플릿의 위치를 가져옵니다.
    POSITION pos = pApp->GetFirstDocTemplatePosition();
    if (pos == NULL)
        return;

    // 첫 번째 문서 템플릿을 가져옵니다.
    CDocTemplate* pDocTemplate = pApp->GetNextDocTemplate(pos);
    if (pDocTemplate == NULL)
        return;

    // 새로운 프레임 윈도우 생성
    CFrameWnd* pNewFrame = pDocTemplate->CreateNewFrame(GetDocument(), NULL);
    if (pNewFrame == NULL)
        return;

    // 프레임 윈도우 초기화
    pDocTemplate->InitialUpdateFrame(pNewFrame, GetDocument(), TRUE);

    // 프레임 윈도우 표시
    pNewFrame->ShowWindow(SW_SHOW);
    pNewFrame->UpdateWindow();
}
// *********************************** 메뉴바 기능 ***********************************

// *********************************** 캐럿 ***********************************
void CMFCMemoView::OnSetFocus(CWnd* pOldWnd)
{
    CView::OnSetFocus(pOldWnd);
    CreateCaretIfNeeded();
    ShowCaret();
    m_bCaretVisible = true;
}

void CMFCMemoView::OnKillFocus(CWnd* pNewWnd)
{
    CView::OnKillFocus(pNewWnd);
    HideCaret();
    DestroyCaret();
    m_bCaretVisible = false;
}

void CMFCMemoView::CreateCaretIfNeeded()
{
    if (!m_bCaretVisible)
    {
        CreateSolidCaret(1, 20);
        ShowCaret();
        m_bCaretVisible = true;
    }
}

void CMFCMemoView::UpdateCaretPosition()
{
    CClientDC dc(this);
    CFont font;
    font.CreateFontW(20, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Arial");
    CFont* pOldFont = dc.SelectObject(&font);

    int x = dc.GetTextExtent(CString(m_textArray[m_nCurrentRow], m_nCurrentColumn)).cx;
    int y = m_nCurrentRow * 20;

    SetCaretPos(CPoint(x, y));

    dc.SelectObject(pOldFont);
}

void CMFCMemoView::OnInitialUpdate()
{
    SetScrollSizes(MM_TEXT, CSize(m_nScrollWidth, m_nScrollHeight));    // 스크롤 바를 설정합니다.
    CScrollView::OnInitialUpdate();
}
// *********************************** 캐럿 ***********************************

// *********************************** 스크롤 ***********************************
void CMFCMemoView::OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
    // 수직 스크롤 동작 처리
    int scrollPos = GetScrollPosition().y;
    int maxPos = GetScrollLimit(SB_VERT);

    switch (nSBCode)
    {
    case SB_LINEUP:
        scrollPos = max(scrollPos - 20, 0);
        break;
    case SB_LINEDOWN:
        scrollPos = min(scrollPos + 20, maxPos);
        break;
    case SB_PAGEUP:
        scrollPos = max(scrollPos - 100, 0);
        break;
    case SB_PAGEDOWN:
        scrollPos = min(scrollPos + 100, maxPos);
        break;
    case SB_THUMBPOSITION:
    case SB_THUMBTRACK:
        scrollPos = nPos;
        break;
    }
    ScrollToPosition(CPoint(GetScrollPosition().x, scrollPos));
    CScrollView::OnVScroll(nSBCode, nPos, pScrollBar);
}

void CMFCMemoView::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
    // 수평 스크롤 동작 처리
    int scrollPos = GetScrollPosition().x;
    int maxPos = GetScrollLimit(SB_HORZ);

    switch (nSBCode)
    {
    case SB_LINELEFT:
        scrollPos = max(scrollPos - 20, 0);
        break;
    case SB_LINERIGHT:
        scrollPos = min(scrollPos + 20, maxPos);
        break;
    case SB_PAGELEFT:
        scrollPos = max(scrollPos - 100, 0);
        break;
    case SB_PAGERIGHT:
        scrollPos = min(scrollPos + 100, maxPos);
        break;
    case SB_THUMBPOSITION:
    case SB_THUMBTRACK:
        scrollPos = nPos;
        break;
    }
    ScrollToPosition(CPoint(scrollPos, GetScrollPosition().y));
    CScrollView::OnHScroll(nSBCode, nPos, pScrollBar);
}
// *********************************** 스크롤 ***********************************